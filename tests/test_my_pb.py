import fixture
import subprocess
import os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_1():
    circuit_fname = file_relative_to_test('configs/pb.yaml')

    fixture.run(circuit_fname)


if __name__ == '__main__':
    test_1()
