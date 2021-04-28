import fixture
import fault
import magma
from pathlib import Path
import pytest
import os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)


names = [
    'configs/differential_amp.yaml',
    'configs/differential_amp_B.yaml',
    'configs/differential_amp2_B.yaml'
]

@pytest.mark.parametrize('config', names)
def test_by_config(config):
    print('GOT CONFIG', config)
    circuit_fname = file_relative_to_test(config)
    print('GOT FNAME', circuit_fname)
    fixture.run(circuit_fname)

if __name__ == '__main__':
    test_by_config(names[2])
    pass

