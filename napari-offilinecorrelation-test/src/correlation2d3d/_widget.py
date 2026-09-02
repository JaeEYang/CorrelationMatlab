from napari import current_viewer
from magicgui.widgets import Container, Label

from correlation2d3d.offline_correlation import (
    make_offline_correlation_widget,
)

def offline_correlation_widget():
    viewer = current_viewer()

    return make_offline_correlation_widget(
        viewer
    )

def serialem_integration_widget():
    from correlation2d3d.offline_widgets import (
        points2nav_widget,
    )
    viewer = current_viewer()
    return Container(
        widgets=[
            Label(
                value="SerialEM Integration Tools"
            ),
            points2nav_widget(viewer),
        ]
    )