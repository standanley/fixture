import fixture
import subprocess
import os
import shutil
import pytest

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

@pytest.mark.xfail(reason='Nested buses currently unsupported in fault spice_target')
def test_mac():
    circuit_fname = file_relative_to_test('configs/mac1.yaml')
    fixture.run(circuit_fname)


if __name__ == '__main__':
    test_mac()
