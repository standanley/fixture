import fixture
import fault
import magma
from pathlib import Path
import pytest
import os
import shutil

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)


# commented-out tests have not been updated to the new config style
tests = [
    #('configs/differential_amp_fake.yaml', None),
    #('configs/differential_amp.yaml', None),
    ('configs/ctle.yaml', 'hspice'),
    #('configs/ctle_model.yaml', 'irun')
]

@pytest.mark.parametrize('test', tests)
def test_by_config(test):
    config, req = test
    if req is not None and not shutil.which(req):
        pytest.skip(f'Requirement "{req}" not installed here')
        return
    circuit_fname = file_relative_to_test(config)
    print('GOT FNAME', circuit_fname)
    fixture.run(circuit_fname)

if __name__ == '__main__':
    test_by_config(tests[3])
    pass

