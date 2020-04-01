import subprocess, os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_simple_amp():
    circuit_fname = file_relative_to_test('./configs/simple_amp.yaml')
    command = 'python -m fixture.run %s' % (circuit_fname)
    print('about to run!')
    res = subprocess.run(command.split(), check=True)
    print('<<<')
    print(res)
    print('>>>')

def test_simple_amp_direct():
    circuit_fname = file_relative_to_test('./configs/simple_amp.yaml')

    import fixture.run as r
    r(circuit_fname)


def test_parameterized_amp():
    circuit_fname = file_relative_to_test('./configs/parameterized_amp.yaml')
    command = 'python -m fixture.run %s' % (circuit_fname)
    res = subprocess.run(command.split(), check=True)

def test_differential_amp():
    circuit_fname = file_relative_to_test('./configs/differential_amp.yaml')
    command = 'python -m fixture.run %s' % (circuit_fname)
    res = subprocess.run(command.split(), check=True)

def test_differential_amp_B():
    circuit_fname = file_relative_to_test('./configs/differential_amp_B.yaml')
    command = 'python -m fixture.run %s' % (circuit_fname)
    res = subprocess.run(command.split(), check=True)

if __name__ == '__main__':
    #test_simple_amp()
    test_parameterized_amp()
    #test_simple_amp_direct()
    #test_differential_amp()
    #test_differential_amp_B()
