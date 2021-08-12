import math
import itertools
import random
random.seed(4)

class Sampler:
    @classmethod
    def rand(cls):
        return random.random()

    @classmethod
    def get_orthogonal_samples(cls, Da, Dd, N):
        ''' Da is dimension of true analog space
            Dd is dimension of digital with analog intent
            N is the requensted number of samples

            Performs Latin Hypercube Sampling on all ports, 
            and also orthogonal smapling on the true analog subset '''

        points = []
        points_digital = []

        # untaken rows and columns for LHS
        D = Da + (1 if Dd>0 else 0)
        available = [set(range(N)) for d in range(D)]
        digital_dim = Da if Dd>0 else None

        # choose a point on [0,1) while respecting and updating available
        def choose_without_regions(dim):
            choice = int(cls.rand() * len(available[dim]))
            row = list(available[dim])[choice]
            available[dim].remove(row)
            p = (row + cls.rand())/N
            return p

        def choose_thermometer(row):
            eps = 1e-8
            # because we want to round equally on either side
            # of the center, we stretch by (1+eps) to avoid
            # things coinciding exactly
            # therm = int((row+.5)/N * (Dd+1))
            num_ones = int((((row + .5) / N - .5) * (1 + eps) + 0.5) * (Dd + 1))
            ones = random.sample(range(Dd), num_ones)
            dig = [1 if i in ones else 0 for i in range(Dd)]
            return dig

        # now break the analog space in to regions
        # each analog dimension is broken into rpd spaces
        # stands for "regions per (true analog) dimension"
        # All digital dimensions together make up one analog dimension
        rpd = 0 if D == 0 else int(math.pow(N, 1/D))

        # equivalent to Da nested for loops
        for region_coords in itertools.product(range(rpd), repeat=D): # rpd^D <= N
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

                # true analog
                rand_in_avail = cls.rand() * length
                for row, segment in avail:
                    l = segment[1] - segment[0]
                    if rand_in_avail < l:
                        # rand_in_avail falls in this segment
                        if dim == digital_dim:
                            dig = choose_thermometer(row)
                            points_digital.append(dig)
                        else:
                            point.append(segment[0] + rand_in_avail)
                            assert (row/N) <= point[-1] < ((row+1)/N), 'point not in row!' # noqa
                        available[dim].remove(row)
                        break
                    rand_in_avail -= l
                else:
                    # else after a for loop runs iff no break happened
                    assert False, 'rand_in_avail not within avail! bug in sampler.py' # noqa

            # now we can do the digital analog intent stuff without worrying about regions
            # We do this a little strangely to cater to thermometer coded stuff

            points.append(point)

        for _ in range(N - len(points)):
            point = []
            for dim in range(D):
                p = choose_without_regions(dim)
                if dim == digital_dim:
                    # go from point back to row
                    row = int(p*N)
                    dig = choose_thermometer(row)
                    points_digital.append(dig)
                else:
                    point.append(p)

            points.append(point)


        # now we do some special adjustment to the digital bits so that each
        # bit always has the same number of 0s and 1s
        errors = [sum(p[i] for p in points_digital)-N/2 for i in range(Dd)]

        # the general strategy is to swap elements within one point, so the bit
        # count of a point never changes but errors decrease
        def bit_perm():
            xs = list(range(Dd))
            random.shuffle(xs)
            return xs

        def get_swappable(i, over):
            # in point i, find a zero that could be a one or vice versa
            # return the j or None, and False iff this isn't necessary
            # the "necessary" thing is important when N is odd
            value_search = 1 if over else 0
            for j in bit_perm():
                if (((errors[j] > 0.75) if over else (errors[j] < -0.75))
                    and points_digital[i][j] == value_search):
                    return j, True
            for j in bit_perm():
                if (((errors[j] > 0.25) if over else (errors[j] < -0.25))
                    and points_digital[i][j] == value_search):
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
            points_digital[i][over] = 0
            points_digital[i][under] = 1
            errors[over] -= 1
            errors[under] += 1


        # pack analog and digital bits together
        if Dd != 0:
            points_combined = [a+d for a,d in zip(points, points_digital)]
        else:
            points_combined = points

        return points_combined

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








