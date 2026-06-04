import nanofractal


def test_version():
    assert nanofractal.__version__ == "0.1.0"


def test_opencv_linked():
    v = nanofractal._opencv_version()
    assert isinstance(v, str)
    assert len(v) > 0
