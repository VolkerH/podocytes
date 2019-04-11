import os
import time
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('wxagg')
from skimage import io
import pims
import jpype
import matplotlib.pyplot as plt
import pathlib
from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser

from skimage.util import invert
from skimage.filters import threshold_yen, gaussian
from skimage.morphology import ball, watershed, binary_closing, binary_dilation
from skimage.measure import label, regionprops
from skimage.feature import blob_dog

import tifffile._tifffile  # imported to silence pims warning

from podocytes.__init__ import __version__
from podocytes.util import (configure_parser_default,
                            parse_args,
                            log_file_begins,
                            log_file_ends,
                            find_files)
from podocytes.image_processing import (crop_region_of_interest,
                                        denoise_image,
                                        filter_by_size,
                                        find_glomeruli,
                                        find_podocytes,
                                        gradient_of_image,
                                        marker_controlled_watershed,
                                        markers_from_blob_coords)
from podocytes.statistics import (glom_statistics,
                                  podocyte_statistics,
                                  podocyte_avg_statistics,
                                  summarize_statistics)


def main():
    args = configure_parser()
    run_program(args)


def run_program(args):
    time_start = log_file_begins(args)
    timestamp = time.strftime('%d-%b-%Y_%H-%M%p', time.localtime())

    # Get to work
    stats_list = []
    filelist = find_files(args.input_directory, args.file_extension)
    logging.info(f"Java path: {jpype.get_default_jvm_path()}")
    logging.info(f"{len(filelist)} {args.file_extension} files found.")
    for filename in filelist:
        logging.info(f"Processing file: {filename}")
        try:
            images = pims.Bioformats(filename)
        except Exception as err:
            logging.warning(f'Exception raised when trying to open {filename}')
            logging.warning(f'{str(type(err))[8:-2]}: {err}')
            continue  # move on to the next file
        for im_series_num in range(images.metadata.ImageCount()):
            logging.info(f"{images.metadata.ImageID(im_series_num)}")
            logging.info(f"{images.metadata.ImageName(im_series_num)}")
            images.series = im_series_num
            images.bundle_axes = 'zyxc'
            single_image_stats = process_image_series(images, filename, args, outfolder=args.output_directory)
            stats_list.append(single_image_stats)
    # Summarize output and write to file
    try:
        detailed_stats = pd.concat(stats_list, ignore_index=True, copy=False)
    except ValueError as err:
        logging.warning(f'No glomeruli identified in these images.')
        logging.warning(f'{str(type(err))[8:-2]}: {err}')
        return None
    else:
        output_filename_detailed_stats = os.path.join(args.output_directory,
                'Podocyte_detailed_stats_' + timestamp + '.csv')
        output_filename_summary_stats = os.path.join(args.output_directory,
                'Podocyte_summary_stats_' + timestamp + '.csv')
        detailed_stats.to_csv(output_filename_detailed_stats)
        summary_stats = summarize_statistics(detailed_stats,
                                             output_filename_summary_stats)
        if len(summary_stats) > 0:
            total_gloms_counted = len(summary_stats)
        else:
            total_gloms_counted = 0
    log_file_ends(time_start, total_gloms_counted=total_gloms_counted)


__DESCR__ = ('Load, segment, count, and measure glomeruli and podocytes in '
             f'fluorescence images.\nVersion {__version__}')
@gooey(default_size=(800, 700),
       image_dir=os.path.join(os.path.dirname(__file__), 'app-images'),
       navigation='TABBED')
def configure_parser():
    """Configure parser and add user input arguments.

    Returns
    -------
    args : argparse arguments
        Parsed user input arguments.

    """
    parser = GooeyParser(prog='Podocyte Profiler', description=__DESCR__)
    parser = configure_parser_default(parser)
    args = parse_args(parser)
    return args


def process_image_series(images, filename, args, save_intermediates=True, outfolder=None):
    """Process a single image series to count the glomeruli and podocytes.

    Parameters
    ----------
    images : pims image object, where images[0] is the image ndarray.
        Input image plus metadata.
    filename : str
        Input image filename.
    args : user input arguments

    Returns
    -------
    single_image_stats : DataFrame
    glomeruli_labels: np.array

    """
    logging.info(f"process_... {filename}")
    logging.info(f"process_... {outfolder}")

    df_list = []
    glomeruli_view = images[0][..., args.glomeruli_channel_number]
    podocytes_view = images[0][..., args.podocyte_channel_number]
    voxel_volume = images[0].metadata['mpp'] * \
                   images[0].metadata['mpp'] * \
                   images[0].metadata['mppZ']
    logging.info(f"Voxel volume in real space: {voxel_volume}")
    #print(pathlib.Path(outfolder) / (str(pathlib.Path(filename).name) + "_label_glom"+str(i).zfill(3) + ".tif")))
    glomeruli_labels = find_glomeruli(glomeruli_view)
    if save_intermediates:
        nz, ny, nx = glomeruli_view.shape
        logging.info(f"glom region size nz {nz} ny {ny} nx {nx}")
        tmp = np.zeros((nz, ny, nx, 3), dtype=np.uint16)
        tmp[...,0] = glomeruli_view
        tmp[...,1] = podocytes_view
        tmp[...,2] = glomeruli_labels
        fname = pathlib.Path(outfolder) / (str(pathlib.Path(filename).name) + "_label_glom" + ".tif")
        io.imsave(str(fname), tmp)
    glom_regions = filter_by_size(glomeruli_labels,
                                  args.minimum_glomerular_diameter,
                                  args.maximum_glomerular_diameter)
    glom_index = 0  # labels not always sequential after filtering by size
    logging.info(f"{len(glom_regions)} glomeruli identified.")
    if len(glom_regions) > 0:
        logging.info(f"print podocytes dtype before: {podocytes_view.dtype}, {np.max(podocytes_view)}")
        podocytes_view = denoise_image(podocytes_view)
        logging.info(f"print podocytes dtype after: {podocytes_view.dtype}, {np.max(podocytes_view)}")

        for i, glom in enumerate(glom_regions):
            logging.info(f"glom bbox {glom.bbox}")
            podocyte_regions, centroid_offset, wshed, podoim_roi, glomim_roi = \
                    find_podocytes(podocyte_image=podocytes_view, glomeruli_image=glomeruli_view, glomeruli_region=glom)
            if save_intermediates:
                nz, ny, nx = podoim_roi.shape
                logging.info(f"nz {nz} ny {ny} nx {nx}")
                tmp = np.zeros((nz, ny, nx, 3), dtype=np.uint16)
                tmp[...,0] = glomim_roi
                tmp[...,1] = (podoim_roi*255.0).astype(np.uint16)
                print(f"podoim roi max {np.max(podoim_roi)}")
                tmp[...,2] = wshed
                fname =pathlib.Path(outfolder) / (str(pathlib.Path(filename).name) + "_wshed_podo"+str(i).zfill(3) + ".tif")
                io.imsave(str(fname), tmp) 
            df = podocyte_statistics(podocyte_regions,
                                     centroid_offset,
                                     voxel_volume)
            logging.info(f"{len(df)} podocytes found for glomerulus " +
                         f"with centroid voxel coord (x,y,z): (" +
                         f"{int(glom.centroid[2])}, " +
                         f"{int(glom.centroid[1])}, " +
                         f"{int(glom.centroid[0])})")
            if len(df) > 0:
                df = podocyte_avg_statistics(df)
                df = glom_statistics(df, glom, glom_index, voxel_volume)
                df['image_series_num'] = images.metadata.ImageID(images.series)
                df['image_series_name'] = images.metadata.ImageName(images.series)
                df['image_filename'] = filename
                glom_index += 1
                df_list.append(df)
    try:
        single_image_stats = pd.concat(df_list, ignore_index=True, copy=False)
    except ValueError as err:
        logging.warning(f'No glomeruli identified.')
        logging.warning(f'{str(type(err))[8:-2]}: {err}')
    else:
        return single_image_stats


if __name__ == '__main__':
    main()
