from pathlib import Path

from correlation2d3d import offline_widgets


def test_normalize_windows_path_removes_quotes_and_normalizes_slashes():
    assert (
        offline_widgets._normalize_windows_path(r'"X:\RawData\session\map.st"')
        == "X:/RawData/session/map.st"
    )
    assert (
        offline_widgets._normalize_windows_path(r"'X:\RawData\session\map.st'")
        == "X:/RawData/session/map.st"
    )


def test_resolve_map_path_finds_stored_path(tmp_path: Path):
    nav_path = tmp_path / "navigator" / "session.nav"
    nav_path.parent.mkdir()
    nav_path.touch()

    map_path = tmp_path / "acquisition" / "map.st"
    map_path.parent.mkdir()
    map_path.touch()

    resolved = offline_widgets._resolve_map_path(str(nav_path), str(map_path))

    assert resolved == map_path


def test_resolve_map_path_finds_filename_beside_nav(tmp_path: Path):
    nav_path = tmp_path / "session" / "session.nav"
    nav_path.parent.mkdir()
    nav_path.touch()

    map_path = nav_path.parent / "map.st"
    map_path.touch()

    resolved = offline_widgets._resolve_map_path(
        str(nav_path),
        r"X:\AcquisitionMachine\session\map.st",
    )

    assert resolved == map_path


def test_resolve_map_path_returns_none_when_map_is_missing(tmp_path: Path):
    nav_path = tmp_path / "session.nav"
    nav_path.touch()

    resolved = offline_widgets._resolve_map_path(
        str(nav_path),
        r"X:\AcquisitionMachine\session\missing.st",
    )

    assert resolved is None
