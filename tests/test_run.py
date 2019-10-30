import subprocess, os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_simple_amp():
    fname = file_relative_to_test('./configs/simple_amp.yaml')
    command = 'python -m fixture.run %s' % fname 
    print('about to run!')
    res = subprocess.run(command.split(), check=True)
    print('<<<')
    print(res)
    print('>>>')

def test_parameterized_amp():
    fname = file_relative_to_test('./configs/parameterized_amp.yaml')
    command = 'python -m fixture.run %s' % fname 
    res = subprocess.run(command.split(), check=True)

if __name__ == '__main__':
    #test_simple_amp()
    test_parameterized_amp()
