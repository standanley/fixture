import math
import itertools
import random

class Sampler:
    def rand():
        return random.random()

    @classmethod
    def get_orthogonal_samples(cls, Da, Dd, N):
        ''' Da is dimension of true analog space
            Dd is dimension of digital with analog intent
            N is the requensted number of samples

            Performs Latin Hypercube Sampling on all ports, 
            and also orthogonal smapling on the true analog subset '''

        points = []

        # untaken rows and columns for LHS
        available = [set(range(N)) for d in range(Da+Dd)]

        # choose a point on [0,1) while respecting and updating available
        def choose_without_regions(dim, digital = False):
            choice = int(cls.rand() * len(available[dim]))
            row = list(available[dim])[choice]
            available[dim].remove(row)
            p = (row + cls.rand())/N
            if digital:
                return 0 if p < 0.5 else 1
            else:
                return p

        # now break the true analog space in to regions
        # each analog dimension is broken into rpd spaces
        # stands for "regions per (true analog) dimension"
        rpd = 0 if Da == 0 else int(math.pow(N, 1/Da))

        # equivalent to Da nested for loops
        for region_coords in itertools.product(range(rpd), repeat=Da): # rpd^Da <= N
            # choose a random point within the region
            point = []
            for dim, rc in enumerate(region_coords): # Da
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
                        segment = (max(start, row/N), min(end, (row+1)/N))
                        avail.append((row, segment))
                        length += segment[1] - segment[0]

                if avail == []:
                    print('\n'.join(str(x) for x in points)) 
                    print('avail', avail, length)
                    print(available[dim], start, end)

                rand_in_avail = cls.rand() * length
                for row, segment in avail:
                    l = segment[1] - segment[0]
                    if rand_in_avail < l:
                        # rand_in_avail falls in this segment
                        point.append(segment[0] + rand_in_avail)
                        available[dim].remove(row)
                        assert (row/N) <= point[-1] < ((row+1)/N), 'point not in row!' # noqa
                        break
                    rand_in_avail -= l
                else:
                    # else after a for loop runs iff no break happened
                    assert False, 'rand_in_avail not within avail! bug in sampler.py' # noqa

            # now we can do the digital analog intent stuff without worrying about regions
            for dim in range(Da, Da+Dd):
                p = choose_without_regions(dim, digital=True)
                point.append(p)

            points.append(point)

        for _ in range(N - len(points)):
            point = []
            for dim in range(Da+Dd):
                p = choose_without_regions(dim, digital = (dim >= Da))
                point.append(p)
            points.append(point)
        return points

    def assert_lhs(samples):
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
            print(bits)
            zeros = len([bit for bit in bits if bit == 0])
            N = len(bits)
            print(bits, N)
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
    samples = Sampler.get_orthogonal_samples(Da, Dd, 5)

    reorg = list(zip(*samples))
    analog_samples = list(zip(*(reorg[:Da])))
    print(analog_samples)
    digital_samples = list(zip(*(reorg[Da:])))

    Sampler.assert_fifty_fifty(digital_samples)
    Sampler.assert_lhs(analog_samples)
    visualize(samples)


    









