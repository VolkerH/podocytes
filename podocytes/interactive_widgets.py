import ipywidgets as widgets

method_select = widgets.Select(
    options=['distance', 'hmax'],
    value='distance',
    description='Method',
    disabled=False
)

hval_slider = widgets.FloatSlider(
    value=1,
    min=0.1,
    max=60,
    step=0.1,
    description='hval threshold:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
    layout = widgets.Layout(width='800px')
)


thresh_slider = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=2.0,
    step=0.05,
    description='Thresh. adjust:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
    layout = widgets.Layout(width='800px')
)

vol_range = widgets.IntRangeSlider(
    value=[4, 1200],
    min=0,
    max=3000,
    step=1,
    description='Filter #Voxels:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    layout = widgets.Layout(width='800px')
)
