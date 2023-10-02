import fixture
import pytest
import os
import shutil

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)


# Commented out things are likely not yet updated to the new config style
tests = [
    ('configs/LDO_clocked_comparator.yaml', None),
]

@pytest.mark.parametrize('test', tests)
def test_by_config(test):
    config, req = test
    print('Testing ', config)
    if req is not None and not shutil.which(req):
        pytest.skip(f'Requirement "{req}" not installed here')
        return
    circuit_fname = file_relative_to_test(config)
    print('GOT FNAME', circuit_fname)
    fixture.run(circuit_fname)

if __name__ == '__main__':
    test_by_config(tests[0])

