# collecting a whole lot of stuff I am using in the notebook
from skimage import img_as_uint, img_as_ubyte
from skimage.morphology import erosion, dilation, ball, disk, cube, square, watershed
from skimage.filters import gaussian, threshold_yen, threshold_otsu
from skimage.morphology import ball,  binary_closing, binary_dilation
from skimage.measure import label, regionprops

from gputools import blur
import matplotlib.pyplot as plt
import logging
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology.extrema import h_maxima, local_maxima

import matplotlib

tmp = np.random.rand(256,3)
tmp[0,:] = (1, 1, 1)
rcmap = matplotlib.colors.ListedColormap(tmp)


# Morphological gradient
def morphograd(array, size=3, round=True):
    if array.ndim == 3:
        if round:
            selem = ball(size)
        else:
            selem = cube(size)
    elif array.ndim == 2:
        if round:
            selem = disk(size)
        else:
            selem = square(size)
    else:
        raise(ValueError, "array has unsupported number of dimensions")
    return dilation(array, selem) - erosion(array, selem)


def smooth(array, sigma, gpu=True):
    if gpu:
        return blur(array, sigma)
    else:
        return gaussian(array, sigma)
 

def label_edges_3d(vol, edgelabel=0):
    vol[0, ...] = edgelabel
    vol[..., 0] = edgelabel
    vol[:, 0, :] = edgelabel
    vol[-1, ...] = edgelabel
    vol[..., -1] = edgelabel
    vol[:, -1, :] = edgelabel
    return(vol)

def remove_labels_by_vol(label_image, min_vol=6, max_vol=400):
    for region in regionprops(label_image):
        if region.area <= min_vol or region.area >= max_vol:
            print(f"removing label {region.label} with volume {region.area}")
            label_image[label_image == region.label] = 0
    return label_image

def find_podocyte_labels(vol, min_dist=4, smoothby=4, gradsmooth=2, graddist=3, visualize=True, h_val=0.3, method='distance', thresh_adjust=0.9, max_vol = 1000):
    # smooth volume
    sm_pod = smooth(vol, smoothby)
    # gradient
    gr_pod = morphograd(smooth(vol, gradsmooth), graddist)

    if method == 'hmax':
        # find seed points for watershed
        maxima = h_maxima(sm_pod, h_val)
        seeds = label(maxima)
    elif method == 'distance':
        # find threshold and apply
        pix_above_zero = sm_pod[sm_pod>0]
        print(threshold_yen(sm_pod))
        print(threshold_yen(pix_above_zero))
        th = thresh_adjust * threshold_yen(pix_above_zero)
        print(f"adjusted threshold : {th}")
        foreground = (sm_pod > th)
        if visualize:
            plt.imshow(np.sum(foreground,axis=0), vmax=3)
            plt.show()
        # split using watershed and distance transform
        distance = distance_transform_edt(foreground)
        maxima = local_maxima(distance, allow_borders=False)
        seeds = label(maxima)
    else:
        raise(ValueError, f"unknown method {method}")
    print(f"found {np.unique(seeds)} seed points")
    seeds = label_edges_3d(seeds)
    podocyte_labels = watershed(gr_pod, seeds)
    print(f"found {np.unique(podocyte_labels)} podocytes") 
    print(f"found {np.unique(np.max(podocyte_labels, axis=0))} podocyte labels in projection")
    filtered_podo = remove_labels_by_vol(podocyte_labels.copy(), max_vol=max_vol)
    print(f"found {np.unique(np.max(filtered_podo))} filtered podocyte labels in projection")

    if visualize:
        plt.imshow(np.max(seeds, axis=0), cmap=rcmap)
        plt.show()
        plt.imshow(np.max(podocyte_labels, axis=0), cmap=rcmap)
        plt.show()        
        plt.imshow(np.max(filtered_podo, axis=0), cmap=rcmap)
        plt.show()
        plt.imshow(np.max(vol, axis=0))
        plt.show()
    return podocyte_labels


def translocation_from_centroids(seg_stk, scale_factor, return_centroids=False):
    """Help on translocation_from_centroids:
    
    translocation_from_centroids(seg_stk, scale_factor, return_centroids=False)
    
        Calculate by how many pixels the centroids of each segmented cell in 
        the seg_stk array must be shifted in x, y and z in order to expand the
        tissue by scale_factor.
        
        Parameters
        ----------
        seg_stk: numpy array (3D or 4D)
            Array to be expanded. If 3D, the array should contain a labeled
            segmentation only. If 4D, dimension 0 should be the channels, with
            the first channel containing the segmentation.      
        scale_factor: int or float
            Factor by which the tissue will be expanded. For example, an input 
            array of shape (100,100,100) with a scale_factor of 1.3 will result
            in an output array of shape (130,130,130). Note that the output
            array is a factor 2 larger than the input; higher scale factors
            will therefore rapidly increase output array size, which may lead
            to memory issues!
        return_centroids: bool (default: False)
            If true, the centroid coordinates, both in the old and the new 
            (shifted) array will be returned. This can be used to make further
            modifications as needed before translocating the pixels.
            
        Returns
        -------
        cell_labels: numpy array (1D)
            Enumeration of the labels of all labeled entities in the seg_stk
            array. Useful to iterate over all cells.
        translocation: numpy array (Nx3)
            Array containing the number of pixels each of the N cell centroids 
            needs to be shifted in x, y and z.
        old_centroids: numpy array (Nx3)
            Only returned if return_centroids==True. Pixel coordinates of the
            centroids for each of the N cells in the unexpanded condition.
        new_centroids: numpy array (Nx3)
            Only returned if return_centroids==True. Pixel coordinates of the
            centroids for each of the N cells in the new, expanded condition.
            
        Requires
        --------
        NumPy 1.9 or similar
        SciPy 0.15 or similar
    """
        
    # Remove channel dimension for 4D arrays; work on segmentation only
    if len(seg_stk.shape) == 4:
        seg_stk = seg_stk[0,:,:,:]
    
    # Create list of cell labels [assume 0 to be background]
    cell_labels = np.unique(seg_stk)[1:]    
    
    # Grab centroids
    from scipy.ndimage import measurements
    old_centroids = np.array(measurements.center_of_mass(seg_stk,labels=seg_stk,index=cell_labels))
    
    # Calculate new centroid positions using scale_factor
    new_centroids = old_centroids * scale_factor

    # Calculate centroid translocation; all pixels belonging to each centroid
    #   will be translocated by the appropriate amount
    translocation = (new_centroids - old_centroids).astype(np.int32)

    # Return results
    if return_centroids:
        return cell_labels, translocation, old_centroids, new_centroids
    else:
        return cell_labels, translocation




# Jonas Hartmanns stuff below
#------------------------------------------------------------------------------

# FUNCTION TO TRANSLOCATE ALL CELL PIXELS

def translocate_pixels(seg_stk, scale_factor, cell_labels, translocation, be_verbose=False):
    """Help on translocate_pixels:
    
    translocate_pixels(seg_stk, scale_factor, cell_labels, translocation, be_verbose=False)
    
        Returns the expanded array ripped_seg where all labeled entities have
        been shifted apart by scale_factor. The shift is done based on pixel
        translocation values which are calculated using the cell's centroids
        (see function translocation_from_centroids).
        
        Parameters
        ----------
        seg_stk: numpy array (3D or 4D)
            Array to be expanded. If 3D, the array should contain a labeled
            segmentation only. If 4D, dimension 0 should be the channels, with
            the first channel containing the segmentation.      
        scale_factor: int or float
            Factor by which the tissue will be expanded. For example, an input 
            array of shape (100,100,100) with a scale_factor of 1.3 will result
            in an output array of shape (130,130,130). Note that the output
            array is a factor 2 larger than the input; higher scale factors
            will therefore rapidly increase output array size, which may lead
            to memory issues!
        cell_labels: numpy array (1D)
            Enumeration of the labels of all labeled entities in the seg_stk
            array. Used to iterate over all cells.
        translocation: numpy array (Nx3)
            Array containing the number of pixels each of the N centroids needs
            to be shifted in x, y and z to achieve an expansion by scale_factor.
        be_verbose: bool (default: False)
            If true, more progress information about the running algorithm will
            be printed.
            
        Returns
        -------
        ripped_seg: numpy array (3D or 4D)
            Expanded ("ripped") input array with the same dimensions and 
            channel order.
        
        Requires
        --------
        NumPy 1.9 or similar
    """
    
    # Warn in case of high bitdepth
    if not (type(seg_stk.flatten()[0]) == np.uint8 or type(seg_stk.flatten()[0]) == np.int8):
        warn('It is recommended to use 8-bit arrays (np.uint8) to minimize memory consumption!')
    
    # Add 1-size dimension to seg_stk for 3D arrays
    if len(seg_stk.shape) == 3:
        seg_stk = np.expand_dims(seg_stk,0)
        
    # Initialize output segmentation array
    stk_dim = np.shape(seg_stk)
    ripped_seg = np.zeros((stk_dim[0],np.int(stk_dim[1]*scale_factor),np.int(stk_dim[2]*scale_factor),np.int(stk_dim[3]*scale_factor)),dtype=type(seg_stk.flatten()[0]))
    
    # Be verbose about progress
    if be_verbose:
        print("\nStarting loop...")
    
    # For each cell...
    for centroid_id,cell_id in enumerate(cell_labels):
    
        # Find all pixel positions in old stack
        old_pxl_indices = np.array(np.where(seg_stk[0,:,:,:]==cell_id))
        
        # Calculate new pixel positions using (broadcasted) centroid translocation
        cell_transloc = translocation[centroid_id].repeat(np.shape(old_pxl_indices)[1]).reshape(3,np.shape(old_pxl_indices)[1])
        new_pxl_indices = old_pxl_indices + cell_transloc
        
        # Assign actual values to the new pixels
        for channel in range(np.shape(seg_stk)[0]):      # For each channel...
            ripped_seg[channel,new_pxl_indices[0],new_pxl_indices[1],new_pxl_indices[2]] = seg_stk[channel,old_pxl_indices[0],old_pxl_indices[1],old_pxl_indices[2]]
        
        # Be verbose about progress
        if be_verbose:
            print("  Done with loop", centroid_id+1, "of", len(cell_labels))
    
    # Be verbose about progress
    if be_verbose:    
        print("Done with loop!")
    
    # Return results
    return ripped_seg
    

#------------------------------------------------------------------------------

# ASSEMBLED TISSUE "RIPPING" FUNCTION

def rip_tissue(seg_stk, scale_factor, be_verbose=False):
    """Help on rip_tissue:
    
    rip_tissue(seg_stk, scale_factor, be_verbose=False)
    
        Expand a segmented tissue by scale_factor, shifting apart the cells but
        preserving cell size, shape and overall tissue organization.
        
        Parameters
        ----------
        seg_stk: numpy array (3D or 4D)
            Array to be expanded. If 3D, the array should contain a labeled
            segmentation only. If 4D, dimension 0 should be the channels, with
            the first channel containing the segmentation.      
        scale_factor: int or float
            Factor by which the tissue will be expanded. For example, an input 
            array of shape (100,100,100) with a scale_factor of 1.3 will result
            in an output array of shape (130,130,130). Note that the output
            array is a factor 2 larger than the input; higher scale factors
            will therefore rapidly increase output array size, which may lead
            to memory issues!
        be_verbose: bool (default: False)
            If true, more progress information about the running algorithm will
            be printed.
            
        Returns
        -------
        ripped_seg: numpy array (3D or 4D)
            Expanded ("ripped") input array with the same dimensions and 
            channel order.
        
        Requires
        --------
        NumPy 1.9 or similar
        SciPy 0.15 or similar
    """
        
    # Be verbose about progress
    if be_verbose:
        print('\nCalculating translocations based on centroids')
        
    # Calculate translocation based on centroids
    cell_labels, translocation = translocation_from_centroids(seg_stk, scale_factor)
    
    # Be verbose about progress
    if be_verbose:
        print('\nTranslocating pixels')
        
    # Translocate all cell pixels
    ripped_seg = translocate_pixels(seg_stk, scale_factor, cell_labels, translocation, be_verbose=be_verbose)
    
    # Be verbose about progress
    if be_verbose:
        print('\nAll done; returning results')
        
    # Return results
    return ripped_seg
