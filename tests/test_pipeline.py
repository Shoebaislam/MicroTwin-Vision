import numpy as np
import pytest
from run_experiment import features, target, key


def test_phase_geometry():
    im = np.zeros((64,64), dtype=np.uint8)
    im[:, :32] = 255
    f = features(im, True)
    assert f[0] == pytest.approx(.5)
    assert f[1] == pytest.approx(1)
    assert f[2] == 1.0
    assert f[3] > 0


def test_label_parsing():
    assert target(b'1.25\n2.3\n',0) == pytest.approx(1.25)
    with pytest.raises(ValueError): target(b'header\n2.0',0)


def test_ids():
    assert key('images/something_0032.tif') == key('diffusion/file_32.txt')
