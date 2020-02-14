import subprocess, os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_command_line():
    circuit_fname = file_relative_to_test('./phase_blender.yaml')
    test_fname = file_relative_to_test('./spectre_spf.yaml')

    command = 'python -m fixture.run %s %s' % (circuit_fname, test_fname)
    print('about to run!')
    res = subprocess.run(command.split(), check=True)
    print('<<<')
    print(res)
    print('>>>')

def test_pb():
    circuit_fname = file_relative_to_test('./phase_blender.yaml')
    test_fname = file_relative_to_test('./spectre_spf.yaml')

    import fixture.run as r
    r(circuit_fname, test_fname)

if __name__ == '__main__':
    test_pb()
    #test_command_line()
