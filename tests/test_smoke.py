from importlib.metadata import version

import nanofractal


def test_version():
    # Match the installed package metadata rather than a hardcoded string so a
    # version bump in pyproject.toml does not require editing this test.
    assert nanofractal.__version__ == version("nanofractal")


def test_opencv_linked():
    v = nanofractal._opencv_version()
    assert isinstance(v, str)
    assert len(v) > 0
