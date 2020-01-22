import subprocess, os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_vco():
    circuit_fname = file_relative_to_test('./configs/vco.yaml')
    test_fname = file_relative_to_test('./configs/ngspice.yaml')

    import fixture.run as r
    r(circuit_fname, test_fname)


if __name__ == '__main__':
    test_vco()
