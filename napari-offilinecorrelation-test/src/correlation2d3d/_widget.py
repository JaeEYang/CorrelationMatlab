from napari import current_viewer
from magicgui.widgets import Container, Label
from .offline_widgets import make_image_panel, load_points_widget

def offline_correlation_widget():
    viewer = current_viewer()
    return Container(widgets=[
        Label(value="<h1>Offline Correlation</h1>"),
        make_image_panel(viewer, "Image 1"),
        make_image_panel(viewer, "Image 2"),
        load_points_widget(viewer),
    ])

def serialem_integration_widget():
    viewer = current_viewer()
    return Container(widgets=[
        Label(value="SerialEM Integration Tools (coming soon)"),
    ])
