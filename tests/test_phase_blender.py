import fixture
import subprocess
import os
import shutil
import pytest

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_ngspice():
    circuit_fname = file_relative_to_test('configs/pb.yaml')

    fixture.run(circuit_fname)

@pytest.mark.skipif(not shutil.which('spectre'), reason='spectre not installed')
def test_spectre():
    circuit_fname = file_relative_to_test('configs/pb1.yaml')

    fixture.run(circuit_fname)

def test_generated():
    circuit_fname = file_relative_to_test('configs/generated_pb.yaml')

    fixture.run(circuit_fname)

if __name__ == '__main__':
    test_generated()
    #test_ngspice()
