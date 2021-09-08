import math
import itertools
import random
from fixture.signals import SignalIn, SignalArray
random.seed(4)

class Sampler:
    @classmethod
    def rand(cls):
        return random.random()

    @classmethod
    def get_samples(cls, dims, N):
        # return a dictionary where keys are SignalIn (each SignalArray in dims
        # will be broken out) and values are length N lists of scaled samples

        samples = cls.get_orthogonal_samples(len(dims), N)
        samples_dict = {}
        for i, dim in enumerate(dims):
            if isinstance(dim, SignalArray):
                bus_type = dim.info.get('bus_type', 'any')

                if bus_type == 'any':
                    # We actually choose numbers of bits to turn on according
                    # to evenly distributed thermometer codes, BUT don't turn
                    # them on in order like we do for thermometer
                    assert len(dim.shape) == 1
                    samples_this_dim = [samples[j][i] for j in range(N)]
                    data = cls.convert_qa_therm_random(samples_this_dim, dim.shape[0])
                    for j, s in enumerate(dim):
                        samples_this_bit = [data[k][j] for k in range(N)]
                        samples_dict[s] = samples_this_bit

                elif bus_type == 'thermometer':
                    assert len(dim.shape) == 1
                    samples_this_dim = [samples[j][i] for j in range(N)]
                    data = cls.convert_qa_therm(samples_this_dim, dim.shape[0])
                    for j, s in enumerate(dim):
                        samples_this_bit = [data[k][j] for k in range(N)]
                        samples_dict[s] = samples_this_bit

                elif bus_type == 'binary':
                    # TODO could probably make this a little better
                    # For now, do nothing special, although we could try to
                    # balance the number of 1s and 0s on a particular bit, etc.
                    samples_this_dim = [samples[j][i] for j in range(N)]
                    data = cls.convert_qa_binary(samples_this_dim, dim.shape[0])
                    for j, s in enumerate(dim):
                        samples_this_bit = [data[k][j] for k in range(N)]
                        samples_dict[s] = samples_this_bit

                else:
                    assert False, f'Unknown bus type {bus_type}'
            else:
                assert isinstance(dim, SignalIn)
                assert dim.type_ in ['analog', 'real']
                assert isinstance(dim.value, tuple) and len(dim.value) == 2
                lims = dim.value
                xs = []
                for j in range(N):
                    xs.append(lims[0] + samples[j][i]*(lims[1]-lims[0]))
                samples_dict[dim] = xs

        return samples_dict

    @classmethod
    def get_orthogonal_samples(cls, D, N):
        '''
        :param D: Number of true analog dimensions
        :param N: Number of samples
        :return: NxD array of samples, every entry between 0 and 1
        Does Latin Hypercube Sampling and Orthogonal sampling
        '''

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

        # now we do some special adjustment to the digital bits so that each
        # bit always has the same number of 0s and 1s
        errors = [sum(column) - N/2 for column in data]
        assert sum(errors) < 0.01

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
    def convert_qa_binary(cls, samples, num_bits):
        data = []
        N = 2**num_bits
        for x in samples:
            v = int(x * N)
            # kinda ugly to get a variable value into a format specifier
            b = f'{{:0{num_bits}b}}'.format(v)
            #b = bin(v)[2:]
            bits = [0 if c == '0' else 1 for c in b]
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

    def assert_fifty_fifty(samples):
        for dim, bits in enumerate(zip(*samples)):
            zeros = len([bit for bit in bits if bit == 0])
            N = len(bits)
            assert abs(zeros - N/2) < 0.75, f'{zeros} out of {N} bits are 0 in dim {dim}'



def visualize(samples):

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
    plt.show()


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








