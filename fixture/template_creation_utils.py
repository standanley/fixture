import random


def poke_binary_analog(tester, port, value):
    '''
    Assuming port is binary analog with port[0] as the
    low-order bit, poke "value" assuming value is scaled
    0 to 1.
    Note that if value is random, this just assigns a random value to each
    port independently, which is what we want to do for testing. Actually
    using "value" is helpful if the port happens to be binary-weighted
    and the value chosen with LHS or orthogonal sampling
    '''

    bits = len(port)
    int_value = int(value * 2**bits)
    for i in range(bits):
        tester.poke(port[i], int_value & 1)
        int_value >>= 1

def plot(xs, ys):
    import matplotlib.pyplot as plt
    plt.plot(xs, ys, '-+')
    plt.grid()
    plt.show()

def debug(test):
    class DebugTest(test):

        def debug(self, tester, port, duration):
            r = tester.get_value(port, params={'style':'block', 'duration': duration})
            self.debug_dict.append((port, r))

        def testbench(self, *args, **kwargs):
            self.debug_dict = []
            print('CREATED DEBUG DICT')
            retval = super().testbench(*args, **kwargs)
            return (self.debug_dict, retval)

        def analysis(self, reads):
            debug_dict, reads_orig = reads

            print('Doing analysis now')

            import matplotlib.pyplot as plt
            leg = []
            for p, r in debug_dict:
                leg.append(self.template.get_name_template(p))
                plt.plot(r.value[0], r.value[1], '-+')
            plt.grid()
            plt.legend(leg)
            plt.show()

            return super().analysis(reads_orig)

    return DebugTest

def make_nondecreasing(ys):
    '''
    Given a list of y values, give a new list of y values that is nondecreasing
    such that the MSE between the two lists is minimized
    '''
    ys = [float(y) for y in ys]
    print('Top of invert function, range for ys', min(ys), max(ys))
    new_ys = [ys[0]]
    for i in range(1, len(ys)):
        #print('i is', i)
        #prev = ys[i-1]
        y = ys[i]
        new_ys.append(y)
        if new_ys[-1] < new_ys[-2]:
            # do some approximating
            # The best thing to do is to take the previous n points and set
            # them equal to their collective average, for some n
            cum_sum = y
            count = 1
            error_added_best = float('inf')
            avg_prev = None
            j = i
            while j > 0:
                j -= 1
                #print('i,j are', i, j)
                cum_sum += ys[j]
                count += 1
                avg = cum_sum / count
                # this time through the loop, we are proposing to set all
                # new_ys[j:i+1] equal to avg (that's j to i, inclusive)
                if j>0 and avg < new_ys[j-1]:
                    # this is illegal because it would be decreasing
                    continue
                error_added = 0
                # loop over all indices we propose to change
                for k in range(j, i+1):
                    error_added += (ys[k] - avg)**2 - (ys[k]-new_ys[k])**2
                if error_added >= error_added_best:
                    # we've gone too far! quit now and use our current best
                    break
                error_added_best = error_added
                avg_best = avg
                j_best = j

            # now that we're out of the loop, we have the j and avg we want
            print('Setting new avg from ', j_best, 'to', i, avg_best)
            for k in range(j_best, i+1):
                new_ys[k] = avg_best

    print('Bottom of invert function, range for ys', min(ys), max(ys))
    # I think we've got new_ys all happy now
    print('DONE', len(ys), len(new_ys))

    import matplotlib.pyplot as plt
    xs = list(range(len(ys)))
    plt.plot(xs, ys, '+')
    plt.plot(xs, new_ys, '--')
    plt.grid()
    plt.show()
    return new_ys



def invert_function(xs, ys):
    ys = [float(y) + random.random()*0.02 for y in ys]
    xs = [float(x) for x in xs]
    xs = list(range(len(xs)))
    # TODO: this is broken for decreasing things
    if ys[0] > ys[-1]:
        temp = [-y for y in ys]
        temp2 = make_nondecreasing(temp)
        ys_up = [-y for y in temp2]
    else:
        ys_up = make_nondecreasing(ys)
    
    new_xs = [xs[0]]
    new_ys = [ys_up[0]]
    float_eps = 1e-10
    for i in range(1, len(xs)-1):
        # look for flat regions
        # start
        if ys_up[i-1] < ys_up[i] - float_eps:
            if ys_up[i] < ys_up[i+1] - float_eps:
                # normal
                new_xs.append(xs[i])
                new_ys.append(ys_up[i])
                print('normal', i)
            else:
                # start of flat region
                frac = (ys_up[i] - ys[i-1]) / (ys[i] - ys[i-1])
                new_x = xs[i-1] + frac * (xs[i] - xs[i-1])
                new_xs.append(new_x)
                new_ys.append(ys_up[i])
                print('start of flat', i, frac, xs[i-1:i+2], new_x)
        else:
            if ys_up[i] < ys_up[i+1] - float_eps:
                # end of flat region
                frac = (ys_up[i] - ys[i]) / (ys[i+1] - ys[i])
                new_x = xs[i] + frac * (xs[i+1] - xs[i])
                new_xs.append(new_x)
                new_ys.append(ys_up[i])
                print('end of flat', i, frac, xs[i-1:i+2], new_x)
            else:
                # middle of flat region - no points necessary
                print('middle of flat', i)
                pass

    new_xs.append(xs[-1])
    new_ys.append(ys_up[-1])

    import matplotlib.pyplot as plt
    plt.plot(xs, ys, '--')
    plt.plot(xs, ys_up, '+')
    plt.plot(new_xs, new_ys, '-x')
    plt.grid()
    plt.show()
    asdf
            
            
        





