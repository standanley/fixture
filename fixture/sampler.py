import math
import itertools
import numbers
import random
import pandas
import numpy as np
from abc import abstractmethod

from fixture.plot_helper import PlotHelper
#from fixture.optional_fit import SampleManager
import fixture.optional_fit
from fixture.signals import SignalIn, SignalArray


class SampleManager:
    SWEEP_ID_TAG = 'sweep_id'
    GROUP_ID_TAG = 'swept_group'
    def __init__(self, optional_groups, test_inputs):
        # TODO how to get optional groups?

        self.optional_groups = []
        assert False, [get_sampler_for_signal(s) for s in optional_groups]
        #rfadj, cfadj = self.optional_groups[0], self.optional_groups[1]
        #def constraint_fun(samples):
        #    x = rfadj
        #    y = cfadj
        #    r = rfadj.get_plot_value(samples)
        #    c = cfadj.get_plot_value(samples)
        #    return abs(r*c - 900) < 100

        #self.optional_groups = [self.optional_groups[2],
        #                        SamplerConstrainted([rfadj, cfadj], constraint_fun)]

        self.test_inputs = [get_sampler_for_signal(ti) for ti in test_inputs]
        self.data = pandas.DataFrame()

    @classmethod
    def sweep_one(cls, test_groups, opt_groups, group, N_test, N_optional):
        # group is the one optional group to sweep, while holding others nominal
        # Choose N_optional values for the optional group, and then for each
        # one sweep N_test different test inputs
        Sampler = fixture.sampler.Sampler
        TEST_DIMS = sum(ti.NUM_DIMS for ti in test_groups)
        new_data = {}

        opt_samples_unscaled = Sampler.get_orthogonal_samples(group.NUM_DIMS, N_optional)
        opt_keys = group.get_nominal().keys()
        opt_samples = {k: [] for k in opt_keys}
        for opt_sample_unscaled in opt_samples_unscaled:
            opt_sample_scaled = group.get(opt_sample_unscaled)
            # make sure the keys match
            assert set(opt_samples) == set(opt_sample_scaled), f'Internal error, maybe keys for {group} get_nominal() dont match the ones for get()'
            for k in opt_sample_scaled:
                opt_samples[k].append(opt_sample_scaled[k])

        '''
        This code works, but the approach is not good - when N_test is small
        you often get really bad coverage of the space just by chance. This
        is especially true when TEST_DIMS > 1 so the orthogonal sampling is 
        less effective
        # rather than do separate test samples for each opt_sample, I want to
        # do them all at once so they are better distributed, and then instead
        # of using those opt-sample values we snap them to a near one
        # from the actual list of opt_sample values
        test_sample_guide = np.array(Sampler.get_orthogonal_samples(
            TEST_DIMS+1,
            N_test*N_optional)
        )
        test_sample_order = test_sample_guide[:, 0].argsort()
        test_samples_flat = test_sample_guide[:, 1:][test_sample_order]
        test_samples = test_samples_flat.reshape((N_optional, N_test, TEST_DIMS))
        '''

        # TODO nominal doesn't necessarily make sense for test inputs
        test_keys = [k for ti in test_groups for k in ti.get_nominal()]
        for k in test_keys:
            new_data[k] = []
        for i in range(N_optional):
            samples_this_opt = Sampler.get_orthogonal_samples(TEST_DIMS, N_test)
            for j in range(N_test):
                target_count = 0
                for ti in test_groups:
                    targets = samples_this_opt[j][target_count : target_count + ti.NUM_DIMS]
                    target_count += ti.NUM_DIMS
                    sample = ti.get(targets)
                    for k in sample:
                        new_data[k].append(sample[k])

        # plot useful for when TEST_DIMS is 2
        #from fixture.plot_helper import plt
        #for i in range(N_optional):
        #    test_set = test_samples[i]
        #    plt.scatter(test_set[:,0], test_set[:,1])
        #plt.grid()
        #plt.show()

        group_ids = []
        sweep_ids = []
        group_data = {k: [] for k in opt_samples}
        for i in range(N_optional):
            group_ids += [group] * N_test
            sweep_ids += [i] * N_test
            for k in opt_samples:
                group_data[k] += [opt_samples[k][i]]*N_test
        new_data[cls.GROUP_ID_TAG] = group_ids
        new_data[cls.SWEEP_ID_TAG] = sweep_ids
        for k in group_data:
            new_data[k] = group_data[k]
        for opt_group in opt_groups:
            if opt_group != group:
                nominal = opt_group.get_nominal()
                for k in nominal:
                    new_data[k] = [nominal[k]] * (N_test * N_optional)

        return pandas.DataFrame(new_data)

    @classmethod
    def sample_all(cls, N, groups_opt, groups_test):
        # TODO I should really add a get_many method to SampleStyle
        groups = groups_opt + groups_test
        NUM_DIMS = sum(g.NUM_DIMS for g in groups)
        raw_samples_list = Sampler.get_orthogonal_samples(NUM_DIMS, N)
        raw_samples = np.array(raw_samples_list)

        data = {}
        dim_count = 0
        for group in groups:
            group_data = raw_samples[:, dim_count:dim_count + group.NUM_DIMS]
            dim_count += group.NUM_DIMS
            group_data_scaled = {k: [] for k in group.get_nominal()}
            for x in group_data:
                datapoint_scaled = group.get(x)
                assert len(datapoint_scaled) == len(group_data_scaled)
                for k in datapoint_scaled:
                    group_data_scaled[k].append(datapoint_scaled[k])
            data.update(group_data_scaled)


        data[cls.GROUP_ID_TAG] = [None]*N
        data[cls.SWEEP_ID_TAG] = [None]*N

        return pandas.DataFrame(data)

class SampleStyle:
    NUM_DIMS = 1

    # MUST have a name and a signals list as well

    @abstractmethod
    def get(self, target):
        # target_samples are between 0 and 1
        # they should be translated to the appropriate space for this input
        # return a dict of {signal: value}
        pass

    @abstractmethod
    def get_nominal(self):
        # return 1 sample at the nominal value
        # return a dict of {signal: value}
        pass

    @abstractmethod
    def get_plot_value(self, sample):
        # given a value of the optional input(s), return the value you would
        # want on a plot axis. Usually this is an identity function, but for
        # arrays of bits it would convert to a decimal value
        pass

    #@abstractmethod
    #def get_dim_names(self, sample):
    #    # return a user-friendly name for each sample dimension
    #    # in most cases this is just the signal.friendly_name()
    #    pass

    def get_many(self, targets):
        # sort of vectorizing self.get(), but instead of returning a list of
        # dictionaries we collate into a dictionary of lists instead
        # Recall that self.get() takes a list, so targets should be 2-dimensional
        ans = {k: [] for k in self.get_nominal()}
        for t in targets:
            for k, v in self.get(t).items():
                ans[k].append(v)
        return ans


    def __str__(self):
        return self.name

class SamplerConst(SampleStyle):
    def __init__(self, signal, value):
        self.signal = signal
        self.signals = [signal]
        self.value = value
        self.name = signal.friendly_name()

    def get(self, target):
        return self.get_nominal()

    def get_nominal(self):
        return {self.signal: self.value}

    def get_plot_value(self, sample):
        # could probably assert sample == self.value
        return sample

class SamplerAnalog(SampleStyle):
    def __init__(self, signal, limits, nominal):
        assert nominal is not None, 'Nominal must be defined before SamplerAnalog'
        assert isinstance(limits, tuple), 'Internal error in config_parse?'
        assert len(limits) == 2, 'Bad limits in SamplerAnalog'
        self.signal = signal
        self.limits = limits
        self.nominal = nominal
        self.signals = [self.signal]
        self.name = signal.friendly_name()

        # TODO this is used by the template writer when creating
        signal.value = self.limits

    def get(self, target):
        assert len(target) == self.NUM_DIMS
        v = self.limits[0] + target[0]*(self.limits[1] - self.limits[0])
        return {self.signal: v}

    def get_nominal(self):
        return {self.signal: self.nominal}

    def get_plot_value(self, sample):
        return sample[self.signal]


class SamplerTestbench(SamplerAnalog):
    # TODO we might be able to get away with a str instead of an IDSignal,
    #  but at least the error message in SamplerAnalog.init would break
    class IDSignal:
        def __init__(self, name):
            self.name = name

        def friendly_name(self):
            return self.name

    def __init__(self, name, limits):
        assert isinstance(limits, tuple), f'limits for SamplerTestbench must be a tuple of length 2 or 3, not a {type(limits)}'
        signal = self.IDSignal(name)
        super().__init__(signal, limits)


class SamplerConstrained(SampleStyle):
    def __init__(self, samplers, constraint_fun):
        #self.signals = signals
        self.samplers = samplers
        self.signals = [s for sampler in samplers for s in sampler.signals]
        self.NUM_DIMS = sum(s.NUM_DIMS for s in self.samplers)
        self.constraint_fun = constraint_fun
        self.constraint_met_count = 0
        self.constraint_failed_count = 0
        self.name = '_'.join(sampler.name for sampler in self.samplers)

    def get(self, target):
        assert len(target) == self.NUM_DIMS
        MIN_SUCCESS_RATE = 1/1e5
        samples = None
        while (self.constraint_met_count + 1) / (self.constraint_met_count + self.constraint_failed_count+1) > MIN_SUCCESS_RATE:
            samples = {}
            i = 0
            for sampler in self.samplers:
                ss = sampler.get(target[i:i+sampler.NUM_DIMS])
                i += sampler.NUM_DIMS
                samples.update(ss)

            if self.constraint_fun(samples):
                # got it!
                self.constraint_met_count += 1
                break
            else:
                self.constraint_failed_count += 1
                target = [random.random() for _ in target]
        else:
            num = self.constraint_met_count
            den = self.constraint_met_count+self.constraint_failed_count
            raise ValueError(f'Issues with rejection sampling, only {num}/{den} successful, latest failure: {samples}')

        return samples

    def get_nominal(self):
        samples = {}
        i = 0
        for sampler in self.samplers:
            ss = sampler.get_nominal()
            i += sampler.NUM_DIMS
            samples.update(ss)
        return samples


class SamplerBinary(SampleStyle):
    def __init__(self, signal):
        assert isinstance(signal, SignalArray)
        #assert signal.bus_info['datatype'] == 'binary_analog'
        assert signal.bus_info.type_ == 'binary', f'Cannot create binary sampler for signal {signal}'
        self.signal = signal
        self.signals = [self.signal]
        #self.first_one = signal.bus_info.first_one
        #if self.first_one is None:

        #    self.first_one = 'low'
        #assert self.first_one in ['low', 'high']
        self.num_bits = len(list(signal))
        self.range_inclusive = signal.value
        if self.range_inclusive is None:
            self.range_inclusive = (0, 2**self.num_bits-1)
        self.name = signal.friendly_name()

    def get_bits(self, v):
        # take the decimal value and return a dict with decimal and bits
        # kinda ugly to get a variable value into a format specifier

        # TODO I'm pretty sure we can use
        #  self.signal.get_decimal_value
        #  but I might need to do a little work to build the dict

        bits = self.signal.get_binary_value(v)
        ans = {s: b for s, b in zip(list(self.signal), bits)}
        ans[self.signal] = v
        return ans

        #b = f'{{:0{self.num_bits}b}}'.format(v)
        #bits = [0 if c == '0' else 1 for c in b]
        #if self.first_one == 'low':
        #    bits = bits[::-1]

        #ans = {self.signal: v}
        #assert len(list(self.signal)) == len(bits)
        #for sig, bit in zip(self.signal, bits):
        #    ans[sig] = bit
        #return ans

    def get(self, target):
        N = self.range_inclusive[1] - self.range_inclusive[0] + 1
        minimum = self.range_inclusive[0]
        v = minimum + int(target[0] * N)
        if v == minimum + N:
            # happens when target is exactly 1
            v -= 1
        return self.get_bits(v)


    def get_nominal(self):
        return self.get_bits(self.signal.nominal)

    def get_plot_value(self, sample):
        # TODO I'm not sure whether the signal is the proper place for this?
        return self.signal.get_decimal_value(sample)


class SamplerTEMP(SampleStyle):
    def __init__(self, name, limits, nominal):
        assert len(limits) == 2, 'Sampling analog requires an input range'
        self.limits = limits
        self.nominal = nominal
        self.signals = None
        self.name = name

    def get(self, target):
        assert len(target) == self.NUM_DIMS
        return self.limits[0] + target[0]*(self.limits[1] - self.limits[0])

    def get_nominal(self):
        return self.nominal

    def get_plot_value(self, sample):
        return {self.name: sample}

    #def get_dim_names(self, sample):
    #    return [self.name]

    def __str__(self):
        return f'SamplerAnalog({self.name})'


def get_sampler_for_signal(signal):
    # TODO should this be done in the config?
    assert isinstance(signal, (SignalIn, SignalArray))
    #assert signal.get_random, 'Can only sample random signals'

    if isinstance(signal, SignalIn):
        if isinstance(signal.value, numbers.Number):
            return [SamplerConst(signal, signal.value)]
        elif isinstance(signal.value, tuple):
            return [SamplerAnalog(signal, signal.value, signal.nominal)]
        else:
            assert False, f'Not sure how to create sample group from value "{signal.value}" for signal {signal.friendly_name()}'

    elif isinstance(signal, SignalArray):
        if signal.bus_info is None:
            # try individual bits?
            ans = []
            for bit in signal:
                ans += get_sampler_for_signal(bit)
            return ans
        else:
            if signal.bus_info.type_ is None:
                assert False, f'To generate stimulus for {signal.friendly_name()} you must define its bus_type in the physical_pin section '
            elif signal.bus_info.type_ == 'binary':
                return [SamplerBinary(signal)]
            elif signal.bus_info.type_ == 'thermometer':
                assert False, 'todo'
            else:
                assert False, f'Cannot create Sampler for bus type {signal.bus_info.type_} from signal {signal}'

    else:
        assert False, f'Can only create sampler from SignalIn or SignalArray, not {type(signal)}'

    #if isinstance(signal, SignalArray):
    #    #assert False, 'TODO'
    #    #return SamplerTEMP(signal.friendly_name(), (0, 63), 32)
    #    return SamplerBinary(signal)
    #else:
    #    assert len(signal.value) == 2, 'Can only sample signal with range'
    #    return SamplerAnalog(signal, signal.value)


class Sampler:
    seed = 4

    @classmethod
    def rand(cls):
        return random.random()

    @classmethod
    def get_samples(cls, test):
        # return a dictionary where keys are SignalIn (each SignalArray in dims
        # will be broken out) and values are length N lists of scaled samples

        # TODO I'm not sure SampleManager does anything useful. Besides its
        #  sweep_one method, it's just a holder for two lists
        data = pandas.DataFrame()
        sm = SampleManager
        for group in test.sample_groups_opt:
            #if not any(s in test.input_signals for s in group.signals):
            new_data = sm.sweep_one(test.sample_groups_test, test.sample_groups_opt, group, 5, 30)
            #new_data = sm.sweep_one(test.sample_groups_test, test.sample_groups_opt, group, 4, 5)
            data = pandas.concat((data, new_data), ignore_index=True)
        data_all = sm.sample_all(100, test.sample_groups_test, test.sample_groups_opt)
        data = pandas.concat((data, data_all), ignore_index=True)

        return data


    @classmethod
    def get_orthogonal_samples(cls, D, N, seed=None):
        '''
        :param D: Number of true analog dimensions
        :param N: Number of samples
        :return: NxD array of samples, every entry between 0 and 1
        Does Latin Hypercube Sampling and Orthogonal sampling
        '''
        # TODO is it okay practice to edit a class property like this?
        if seed is None:
            seed = cls.seed
            cls.seed += 1
        random.seed(seed)

        points = []
        # untaken rows and columns for LHS
        available = [set(range(N)) for _ in range(D)]

        # choose a point on [0,1) while respecting and updating available
        def choose_without_regions(dim):
            choice = int(cls.rand() * len(available[dim]))
            row = list(available[dim])[choice]
            available[dim].remove(row)
            p = (row + cls.rand())/N
            return p


        # now break the analog space into regions
        # each analog dimension is broken into rpd spaces
        # stands for "regions per (true analog) dimension"
        rpd = 0 if D == 0 else int(math.pow(N, 1/D))

        # equivalent to D nested for loops
        # Total number of iterations is rpd^D <= N
        for region_coords in itertools.product(range(rpd), repeat=D):
            # choose a random point within the region
            point = []
            for dim, rc in enumerate(region_coords): # D
                # what sections of this region are still available?
                avail = []
                length = 0
                # start and end of this region in this dimension, entire dim is [0,1)
                start = rc / rpd
                end = (rc+1)/ rpd
                # snap region divisions to nearest row division
                # Be careful! I thought this wan't necessary, but in some cases the greedy
                # approach doesn't find a solution if you don't do this
                start = round(start*N)/N
                end = round(end*N)/N
                # loop through every row overlapping this region
                for row in range(math.floor(start*N), math.ceil(end*N)): # N/rpd <= N
                    if row in available[dim]:
                        #assert max(start, row/N) == row/N
                        #assert min(end, (row+1)/N) == (row+1)/N
                        segment = (max(start, row/N), min(end, (row+1)/N))
                        avail.append((row, segment))
                        length += segment[1] - segment[0]

                if avail == []:
                    print('\n'.join(str(x) for x in points)) 
                    print('avail', avail, length)
                    print(available[dim], start, end)
                    assert False, 'bug in sampler.py'

                # we want to randomly choose a point in theis region in the
                # space still in "available", which keeps track of LHS
                rand_in_avail = cls.rand() * length
                for row, segment in avail:
                    l = segment[1] - segment[0]
                    if rand_in_avail < l:
                        # rand_in_avail falls in this segment
                        point.append(segment[0] + rand_in_avail)
                        assert (row/N) <= point[-1] < ((row+1)/N), 'point not in row!' # noqa
                        available[dim].remove(row)
                        break
                    rand_in_avail -= l
                else:
                    # else after a for loop runs iff no break happened
                    assert False, 'rand_in_avail not within avail! bug in sampler.py' # noqa

            points.append(point)

        # now we have one point in each region, but we need to fill out the rest
        for _ in range(N - len(points)):
            point = []
            for dim in range(D):
                p = choose_without_regions(dim)
                point.append(p)

            points.append(point)

        return points

    @classmethod
    def convert_qa_therm(self, samples, num_bits):
        '''
        :param samples:
        :param num_bits:
        :return:
        We want to turn 0.23432 in to a thermometer code like 11000000,
        so all the 1s are always at the beginning.
        We are actually picky about the total number of 0s and 1s being the
        same, so we choose the number of ones based on the "row" of LHS the
        random number is in, rather than basing it on the random number itself.
        '''
        N = len(samples)

        def choose_thermometer(x):
            row = int(x * N)
            eps = 1e-8
            # because we want to round equally on either side
            # of the center, we stretch by (1+eps) to avoid
            # things coinciding exactly
            # therm = int((row+.5)/N * (nb+1))
            num_ones = int((((row + .5) / N - .5) * (1 + eps) + 0.5) * (num_bits + 1))
            dig = [1 if i < num_ones else 0 for i in range(num_bits)]
            return dig

        data = [choose_thermometer(x) for x in samples]
        return data

    @classmethod
    def convert_qa_therm_random(self, samples, num_bits):
        '''
        :param samples:
        :param num_bits:
        :return:
        We want to turn 0.23432 in to a thermometer code like 00101000, where
        the 1s are randomly distributed throughout the code.
        We are actually picky about the total number of 0s and 1s being the
        same, so we choose the number of ones based on the "row" of LHS the
        random number is in, rather than basing it on the random number itself.
        '''
        N = len(samples)

        def choose_thermometer(x):
            row = int(x * N)
            eps = 1e-8
            # because we want to round equally on either side
            # of the center, we stretch by (1+eps) to avoid
            # things coinciding exactly
            # therm = int((row+.5)/N * (nb+1))
            num_ones = int((((row + .5) / N - .5) * (1 + eps) + 0.5) * (num_bits + 1))
            ones = random.sample(range(num_bits), num_ones)
            dig = [1 if i in ones else 0 for i in range(num_bits)]
            return dig

        data = [choose_thermometer(x) for x in samples]
        data_T = list(zip(*data))

        # now we do some special adjustment to the digital bits so that each
        # bit always has the same number of 0s and 1s
        errors = [sum(column) - N/2 for column in data_T]
        assert abs(sum(errors)) < 0.501

        # the general strategy is to swap elements within one point, so the bit
        # count of a point never changes but errors decrease
        def bit_perm():
            xs = list(range(num_bits))
            random.shuffle(xs)
            return xs

        def get_swappable(i, over):
            # in point i, find a zero that could be a one or vice versa
            # return the j or None, and False iff this isn't necessary
            # the "necessary" thing is important when N is odd
            value_search = 1 if over else 0
            for j in bit_perm():
                if (((errors[j] > 0.75) if over else (errors[j] < -0.75))
                        and data[i][j] == value_search):
                    return j, True
            for j in bit_perm():
                if (((errors[j] > 0.25) if over else (errors[j] < -0.25))
                        and data[i][j] == value_search):
                    return j, False
            return None, False

        options = list(range(N))
        while len(options) > 0:
            opt_choice = random.randint(0, len(options)-1)
            i = options[opt_choice]

            over, over_necessary = get_swappable(i, True)
            under, under_necessary = get_swappable(i, False)

            if (over is None or under is None
                    or (not (over_necessary or under_necessary))):
                # nothing useful at this i
                options.pop(opt_choice)
                continue

            # swap over and under
            data[i][over] = 0
            data[i][under] = 1
            errors[over] -= 1
            errors[under] += 1

        return data

    @classmethod
    def convert_qa_binary(cls, samples, num_bits, first_one,
                          range_inclusive=None):
        assert first_one in ['low', 'high']
        data = []
        if range_inclusive is not None:
            N = range_inclusive[1] - range_inclusive[0] + 1
            minimum = range_inclusive[0]
        else:
            N = 2 ** num_bits
            minimum = 0
        for x in samples:
            v = minimum + int(x * N)
            # kinda ugly to get a variable value into a format specifier
            b = f'{{:0{num_bits}b}}'.format(v)
            # b = bin(v)[2:]
            bits = [0 if c == '0' else 1 for c in b]
            if first_one == 'low':
                bits = bits[::-1]
            data.append(bits)
        return data

    def assert_lhs(samples):
        #visualize([s[0:2] for s in samples])
        # print('samples', samples)
        N = len(samples)
        for dim in range(len(samples[0])):
            for i in range(N):
                # must be at least one sample in this interval
                interval = (i/N, (i+1)/N)
                for j in range(N):
                    if interval[0] <= samples[j][dim] < interval[1]:
                        break
                else:
                    assert False, f'No sample in interval {interval} in dim {dim}'

    @staticmethod
    def assert_fifty_fifty(samples):
        for dim, bits in enumerate(zip(*samples)):
            zeros = len([bit for bit in bits if bit == 0])
            N = len(bits)
            assert abs(zeros - N/2) < 0.75, f'{zeros} out of {N} bits are 0 in dim {dim}'



def visualize(samples):
    from fixture.plot_helper import plt

    #print('\n'.join(str(x) for x in samples))
    import matplotlib.pyplot as plt

    if len(samples[0]) == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        reorg = list(zip(*samples))
        ax.scatter(reorg[0], reorg[1], reorg[2])
        plt.show()
        exit()



    N = len(samples)
    for i in range(N-1):
        z = (i+1)/N
        plt.plot([z, z], [0, 1], color='gray')
        plt.plot([0, 1], [z, z], color='gray')
    M = math.floor(math.pow(N, 1/len([s for s in samples[0] if type(s)==float])))
    for i in range(M-1):
        z = round((i+1)/M*N)/N
        plt.plot([z, z], [0, 1], color='black')
        plt.plot([0, 1], [z, z], color='black')

    reorg = list(zip(*samples))
    plt.plot(*reorg[:2],linestyle='None', marker = 'x')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    #plt.show()
    PlotHelper.save_current_plot('sample_test')
    print('created sample_test plot')


if __name__ == '__main__':
    '''
    for N in range(110, 100, -1):
        print(N)
        for i in range(1000):
            print(i)
            get_orthogonal_samples(2, 0, N)
    '''
    #visualize(get_orthogonal_samples(2, 1, 85))

    Da, Dd = 2, 12
    samples = Sampler.get_orthogonal_samples(Da, Dd, 10)

    reorg = list(zip(*samples))
    analog_samples = list(zip(*(reorg[:Da])))
    print(analog_samples)
    digital_samples = list(zip(*(reorg[Da:])))

    Sampler.assert_fifty_fifty(digital_samples)
    Sampler.assert_lhs(analog_samples)
    visualize(samples)








