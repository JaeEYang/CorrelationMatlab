"""
Offline FLM-TEM Correlation Plugin for napari
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Import existing widgets
from ._widget import ExampleQWidget, example_magic_widget

# Import new FLM-TEM correlation widget
from .correlation_widget import FLMTEMCorrelationWidget

__all__ = (
    "ExampleQWidget",
    "example_magic_widget",
    "FLMTEMCorrelationWidget",  # Add this
)
