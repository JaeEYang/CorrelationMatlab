import subprocess
import sys
import textwrap
def test_headless_packages_do_not_load_gui_stack():
    code = textwrap.dedent(
    """
    import sys

    import correlation2d3d
    import correlation2d3d.core
    import correlation2d3d.fileio

    assert "napari" not in sys.modules
    assert "magicgui" not in sys.modules
    assert "correlation2d3d.offline_widgets" not in sys.modules
    """
)

    subprocess.run([sys.executable, "-c", code], check=True)