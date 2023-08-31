import fixture
import os
import shutil
import pytest

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_ngspice():
    circuit_fname = file_relative_to_test('configs/pb.yaml')

    fixture.run(circuit_fname)

@pytest.mark.skipif(not shutil.which('spectre'), reason='spectre not installed')
def test_spectre():
    circuit_fname = file_relative_to_test('configs/pb1.yaml')

    fixture.run(circuit_fname)

def test_generated():
    circuit_fname = file_relative_to_test('configs/generated_pb.yaml')

    fixture.run(circuit_fname)


@pytest.mark.skipif(not os.path.exists(file_relative_to_test('../sky130/skywater-pdk')),
                    reason='Sky130 not installed')
def test_sky130():
    circuit_fname = file_relative_to_test('configs/pb_sky130.yaml')
    fixture.run(circuit_fname)

@pytest.mark.skipif(not os.path.exists(file_relative_to_test('../sky130/skywater-pdk')),
                    reason='Sky130 not installed')
@pytest.mark.skipif(True,
                    reason='This takes a super long time and I don\'t know why')
def test_C_sky130():
    circuit_fname = file_relative_to_test('configs/pb_C_sky130.yaml')
    fixture.run(circuit_fname)


@pytest.mark.xfail(reason='Bus is currently broken when defined as individual bits.')
@pytest.mark.skipif(not os.path.exists(file_relative_to_test('../sky130/skywater-pdk')),
                    reason='Sky130 not installed')
def test_C_sky130_therm16():
    circuit_fname = file_relative_to_test('configs/pb_C_sky130_therm16.yaml')
    fixture.run(circuit_fname)

@pytest.mark.skipif(not shutil.which('irun'), reason='ncsim not installed')
def test_therm16_model():
    circuit_fname = file_relative_to_test('configs/pb_C_sky130_therm16_model.yaml')
    fixture.run(circuit_fname)


if __name__ == '__main__':
    test_generated()
    #test_ngspice()
    #test_spectre()
    #test_C_sky130()
    #test_C_sky130_therm16()
    #test_therm16_model()
