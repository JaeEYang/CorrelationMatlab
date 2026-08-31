import subprocess
import sys
import textwrap
def test_package_root_does_not_gui_stack():
    code = textwrap.dedent(
    """
    import sys
    import correlation2d3d

    assert "napari" not in sys.modules
    assert "magicgui" not in sys.modules
    assert "correlation2d3d.offline_widgets" not in sys.modules
    """
    )

    subprocess.run([sys.executable, "-c", code], check=True)