import subprocess, os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_simple_amp():
    circuit_fname = file_relative_to_test('./configs/simple_amp.yaml')
    test_fname = file_relative_to_test('./configs/ngspice.yaml')

    command = 'python -m fixture.run %s %s' % (circuit_fname, test_fname)
    print('about to run!')
    res = subprocess.run(command.split(), check=True)
    print('<<<')
    print(res)
    print('>>>')

def test_parameterized_amp():
    circuit_fname = file_relative_to_test('./configs/parameterized_amp.yaml')
    test_fname = file_relative_to_test('./configs/ngspice.yaml')
    command = 'python -m fixture.run %s %s' % (circuit_fname, test_fname)
    res = subprocess.run(command.split(), check=True)

if __name__ == '__main__':
    #test_simple_amp()
    test_parameterized_amp()
