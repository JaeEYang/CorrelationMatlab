import importlib


def test_plugin_package_imports():
    module = importlib.import_module("correlation2d3d")

    assert module is not None
