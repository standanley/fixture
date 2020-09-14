import fixture
import os
import shutil
import pytest

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def test_sampler1():
    circuit_fname = file_relative_to_test('configs/sampler1.yaml')

    fixture.run(circuit_fname)

def test_sampler2():
    circuit_fname = file_relative_to_test('configs/sampler2.yaml')

    fixture.run(circuit_fname)

def test_sampler3():
    circuit_fname = file_relative_to_test('configs/sampler3.yaml')

    fixture.run(circuit_fname)

def test_sampler4():
    circuit_fname = file_relative_to_test('configs/sampler4.yaml')

    fixture.run(circuit_fname)

if __name__ == '__main__':
    test_sampler4()
