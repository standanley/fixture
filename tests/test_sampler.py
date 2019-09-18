from facet import Sampler
import pytest
import random

def transpose(x):
    return list(zip(*x))

def test_assert_fifty_fifty():
    Sampler.assert_fifty_fifty(
            transpose([[0, 0, 1, 1, 1, 0, 0],
                       [1, 0, 0, 0, 1, 1, 1]]))
    Sampler.assert_fifty_fifty(
            transpose([[0, 0, 1, 1, 1, 0],
                       [1, 0, 0, 0, 1, 1]]))
    with pytest.raises(AssertionError):
        Sampler.assert_fifty_fifty(
                transpose([[0, 1, 1, 0, 1, 1],
                           [0, 0, 0, 1, 1, 1]]))
    with pytest.raises(AssertionError):
        Sampler.assert_fifty_fifty(
                transpose([[0, 1, 1, 0, 1, 0],
                           [0, 0, 0, 1, 1, 0]]))

def test_analog():
    for test in range(100):
        d = random.randrange(1, 10)
        N = random.randrange(d+1, 300)
        print(f'Testing {d} dimensions, {N} samples')

        s = Sampler.get_orthogonal_samples(d, 0, N)
        Sampler.assert_lhs(s)

def test_digital():
    for test in range(100):
        d = random.randrange(1, 25)
        N = random.randrange(d+1, 300)
        print(f'Testing {d} dimensions, {N} samples')

        s = Sampler.get_orthogonal_samples(0, d, N)
        Sampler.assert_fifty_fifty(s)

def test_combined():
    for test in range(100):
        Da = random.randrange(1, 10)
        Dd = random.randrange(1, 25)
        N = random.randrange(Da+Dd+1, 300)
        print(f'Testing {Da} analog dimensions, {Dd} digital, {N} samples')

        s = Sampler.get_orthogonal_samples(Da, Dd, N)

        reorg = list(zip(*s))
        analog_samples = list(zip(*(reorg[:Da])))
        digital_samples = list(zip(*(reorg[Da:])))

        Sampler.assert_lhs(analog_samples)
        Sampler.assert_fifty_fifty(digital_samples)

