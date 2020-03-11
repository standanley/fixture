import fixture
import subprocess
import os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_diff():
    circuit_fname = file_relative_to_test('configs/differential_amp_mgenero.yaml')

    fixture.run(circuit_fname)

def test_diff_param():
    circuit_fname = file_relative_to_test('configs/differential_amp_param_mgenero.yaml')

    fixture.run(circuit_fname)

if __name__ == '__main__':
    test_diff_param()
