import fixture
import os
import shutil
import pytest

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_ngspice():
    circuit_fname = file_relative_to_test('configs/sampler1.yaml')

    fixture.run(circuit_fname)


if __name__ == '__main__':
    test_ngspice()
