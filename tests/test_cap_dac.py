import fixture
import os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_cap_dac():
    circuit_fname = file_relative_to_test('configs/cap_dac_nikhil.yaml')
    fixture.run(circuit_fname)


if __name__ == '__main__':
    test_cap_dac()
