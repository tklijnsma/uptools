import pytest
import uptools
import os, os.path as osp

testdir = osp.abspath(osp.dirname(__file__))
testfile = osp.join(testdir, 'ntup.root')

def test_iter_arrays():
    for arrays in uptools.iter_arrays(testfile, decode=True):
        break
    assert uptools.numentries(arrays) == 92
    assert len(arrays['genparticles_pt']) == 92
    with pytest.raises((ValueError, KeyError)):
        arrays[b'genparticles_pt']
