from pathlib import Path

import pytest

from correlation2d3d import offline_widgets


@pytest.fixture(autouse=True)
def clear_session_remaps():
    offline_widgets._map_directory_remaps.clear()
    yield
    offline_widgets._map_directory_remaps.clear()


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


def test_explicit_selection_teaches_sibling_map_directory(tmp_path: Path):
    nav_path = tmp_path / "session.nav"
    nav_path.touch()

    local_maps = tmp_path / "relocated-maps"
    local_maps.mkdir()
    first_map = local_maps / "map-1.st"
    second_map = local_maps / "map-2.st"
    first_map.touch()
    second_map.touch()

    remembered = offline_widgets._remember_map_directory(
        str(nav_path),
        r"X:\AcquisitionMachine\session\map-1.st",
        first_map,
    )
    resolved = offline_widgets._resolve_remembered_map_path(
        str(nav_path),
        r"X:\AcquisitionMachine\session\map-2.st",
    )

    assert remembered is True
    assert resolved == second_map


def test_renamed_selection_is_not_reused_for_sibling_maps(tmp_path: Path):
    nav_path = tmp_path / "session.nav"
    nav_path.touch()

    selected_map = tmp_path / "renamed-map.st"
    selected_map.touch()

    remembered = offline_widgets._remember_map_directory(
        str(nav_path),
        r"X:\AcquisitionMachine\session\original-name.st",
        selected_map,
    )

    assert remembered is False
    assert offline_widgets._map_directory_remaps == {}


def test_absolute_and_bare_mapfile_share_one_cache_key(tmp_path: Path):
    """An absolute MapFile and a bare filename naming the same directory agree.

    The two shapes take different branches of the key builder, so they only
    land on one dictionary entry if both branches canonicalize the separators.
    """
    nav_path = tmp_path / "session.nav"
    nav_path.touch()
    nav_directory = nav_path.resolve().parent

    first_map = nav_directory / "map-1.st"
    second_map = nav_directory / "map-2.st"
    first_map.touch()
    second_map.touch()

    remembered = offline_widgets._remember_map_directory(
        str(nav_path),
        str(first_map),
        first_map,
    )
    resolved = offline_widgets._resolve_remembered_map_path(
        str(nav_path),
        "map-2.st",
    )

    assert remembered is True
    assert resolved == second_map


def test_relative_mapfile_keys_are_anchored_at_the_nav_directory(tmp_path: Path):
    first_nav = tmp_path / "session-a" / "session.nav"
    second_nav = tmp_path / "session-b" / "session.nav"
    for nav_path in (first_nav, second_nav):
        nav_path.parent.mkdir()
        nav_path.touch()

    first_key = offline_widgets._map_source_directory(str(first_nav), r"maps\map.st")
    second_key = offline_widgets._map_source_directory(str(second_nav), r"maps\map.st")

    assert first_key.endswith("session-a/maps")
    assert second_key.endswith("session-b/maps")
    assert first_key != second_key


def test_relative_mapfile_resolves_under_the_nav_directory(tmp_path: Path):
    nav_path = tmp_path / "session.nav"
    nav_path.touch()
    nav_directory = nav_path.resolve().parent

    map_path = nav_directory / "maps" / "map.st"
    map_path.parent.mkdir()
    map_path.touch()

    resolved = offline_widgets._resolve_map_path(str(nav_path), r"maps\map.st")

    assert resolved == map_path


def test_relative_mapfile_relocation_does_not_leak_between_nav_files(tmp_path: Path):
    """A relocation learned for one NAV must not answer a different NAV.

    Both NAV files here use the same relative MapFile text, so an unanchored
    key would collapse them onto one entry and hand the second NAV a map
    belonging to the first.
    """
    first_nav = tmp_path / "session-a" / "session.nav"
    second_nav = tmp_path / "session-b" / "session.nav"
    for nav_path in (first_nav, second_nav):
        nav_path.parent.mkdir()
        nav_path.touch()

    relocated = tmp_path / "relocated-maps"
    relocated.mkdir()
    located_map = relocated / "map.st"
    located_map.touch()

    remembered = offline_widgets._remember_map_directory(
        str(first_nav),
        r"maps\map.st",
        located_map,
    )
    leaked = offline_widgets._resolve_remembered_map_path(
        str(second_nav),
        r"maps\map.st",
    )

    assert remembered is True
    assert leaked is None
