from fixture import config
import os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_pb():
    circuit_filename_relative = 'spice/myphaseblender.sp'
    circuit_filename = file_relative_to_test(circuit_filename_relative)
    text = config.make_config(circuit_filename, 'SimpleAmpTemplate', True)
    print(text)

def test_pb_interactive():
    circuit_filename_relative = 'spice/myphaseblender.sp'
    circuit_filename = file_relative_to_test(circuit_filename_relative)
    text = config.make_config_interactive(circuit_filename, 'PhaseBlenderTemplate_C', True)
    print(text)

if __name__ == '__main__':
    test_pb_interactive()
