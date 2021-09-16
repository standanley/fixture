import math

from fixture import Sampler
import pytest
import random

from fixture.signals import create_input_domain_signal


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
        print(f'Testing {d} analog dimensions, {N} samples')

        s = Sampler.get_orthogonal_samples(d, N)
        assert len(s) == N
        Sampler.assert_lhs(s)

def test_analog_signals():
    for test in range(100):
        d = random.randrange(1, 10)
        N = random.randrange(d+1, 300)
        print(f'Testing {d} analog signals, {N} samples')
        values = []
        for i in range(d):
            a = random.random()*20 - 10
            b = random.random()*20 - 10
            values.append((min(a, b), max(a, b)))

        sigs = [create_input_domain_signal(f'test{i}', v)
                for i, v in enumerate(values)]
        samples = Sampler.get_samples(sigs, N)
        assert len(samples ) == d

        for s, xs in samples.items():
            assert len(xs) == N
            v = s.value
            assert v in values

            lhs_width = (v[1] - v[0]) / N
            assert v[0] <= min(xs) <= v[0] + lhs_width
            assert v[1] - lhs_width <= max(xs) <= v[1]


def test_qa_therm():
    for test in range(100):
        W = random.randrange(1, 25)
        N = random.randrange(W+1, 300)
        print(f'Testing {W}-bit thermometer, {N} samples')

        xs = [x[0] for x in Sampler.get_orthogonal_samples(1, N)]
        bs = Sampler.convert_qa_therm(xs, W)
        assert len(bs) == N
        assert len(bs[0]) == W

        hist = [0]*(W+1)
        for b in bs:
            # should turn on left to right
            zeros = False
            for bb in b:
                if zeros:
                    assert bb == 0
                else:
                    zeros = (bb == 0)

            hist[sum(b)] += 1

        assert sum(hist) == N
        # should be an even distribution over codes
        assert max(hist) - min(hist) <= 1

        # should be centered around the halfway point
        total_ones = sum(count*i for i, count in enumerate(hist))
        assert abs(total_ones*2 - W*N) <= 1

        #for xs in samples.values():
        #    Sampler.assert_fifty_fifty(s)

def test_qa_therm_random():
    # TODO this method shares a lot with the non-random version
    for test in range(100):
        W = random.randrange(1, 25)
        N = random.randrange(W + 1, 300)
        print(f'Testing {W}-bit thermometer, {N} samples')

        xs = [x[0] for x in Sampler.get_orthogonal_samples(1, N)]
        bs = Sampler.convert_qa_therm_random(xs, W)
        assert len(bs) == N
        assert len(bs[0]) == W

        hist = [0] * (W + 1)
        for b in bs:
            # does not need to turn on left to right
            hist[sum(b)] += 1

        assert sum(hist) == N
        # should be an even distribution over codes
        assert max(hist) - min(hist) <= 1

        # should be centered around the halfway point
        total_ones = sum(count * i for i, count in enumerate(hist))
        assert abs(total_ones * 2 - W * N) <= 1

        # each bit should turn on half the time
        Sampler.assert_fifty_fifty(bs)

def test_qa_binary():
    for test in range(100):
        W = random.randrange(1, 12)
        N = random.randrange(W + 1, 1000)
        print(f'Testing {W}-bit thermometer, {N} samples')

        xs = [x[0] for x in Sampler.get_orthogonal_samples(1, N)]
        bs = Sampler.convert_qa_binary(xs, W)
        assert len(bs) == N
        assert len(bs[0]) == W

        hist = [0] * (2**W)
        for b in bs:
            v = int(''.join(str(bb) for bb in b), 2)
            hist[v] += 1

        assert sum(hist) == N
        # should be a ROUGHLY even distribution over codes
        # TODO I don't actually know whether 3 is correct here, but it passed
        # 10**4 trials
        assert max(hist) - min(hist) <= 3

        # should be ROUGHLY centered around the halfway point
        # highest we can get is if every LHS sample is at the top of its slice
        # (2**W-1)/N/2 above the expected value
        total_val = sum(count * i for i, count in enumerate(hist))
        assert abs(total_val*2 - (2**W-1)*N) <= math.ceil((2**W - 1) / (2*N))*N

        # each bit should turn on ROUGHLY half the time
        # TODO check this automatically ... I know it's true right now
        #Sampler.assert_fifty_fifty(bs)

# TODO write a test for this
#def test_combined():
#    for test in range(100):
#        Da = random.randrange(1, 10)
#        Dd = random.randrange(1, 25)
#        N = random.randrange(Da+Dd+1, 300)
#        print(f'Testing {Da} analog dimensions, {Dd} digital, {N} samples')
#
#        s = Sampler.get_orthogonal_samples(Da, Dd, N)
#        assert len(s) == N
#
#        reorg = list(zip(*s))
#        analog_samples = list(zip(*(reorg[:Da])))
#        digital_samples = list(zip(*(reorg[Da:])))
#
#        Sampler.assert_lhs(analog_samples)
#        Sampler.assert_fifty_fifty(digital_samples)

if __name__ == '__main__':
    #test_assert_fifty_fifty()
    #test_analog()
    #test_analog_signals()
    #test_qa_therm()
    test_qa_binary()
