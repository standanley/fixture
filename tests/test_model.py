import fixture
import subprocess
import os
import pytest
import shutil

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

# def test_diff():
#     circuit_fname = file_relative_to_test('configs/differential_amp_mgenero.yaml')
#     test_fname = file_relative_to_test('./configs/ngspice.yaml')
# 
#     fixture.run(circuit_fname, test_fname)

@pytest.mark.xfail
def test_diff_param():
    circuit_fname = file_relative_to_test('configs/differential_model.yaml')
    test_fname = file_relative_to_test('./configs/mlingua.yaml')

    fixture.run(circuit_fname, test_fname)

@pytest.mark.skipif(not shutil.which('ncsim'), reason='ncsim not installed')
def test_pb():
    circuit_fname = file_relative_to_test('configs/pb_model.yaml')
    fixture.run(circuit_fname)

if __name__ == '__main__':
    test_pb()
