import random
from scipy.interpolate import interp1d
from fixture import template_master
import fixture.modal_analysis as modal_analysis
import numpy as np

def plot(x, y, legend = None):
    print('called to plot')
    import matplotlib.pyplot as plt
    if type(y) != tuple:
        y = (y,)
    for y in y:
        plt.plot(x, y, '-*')
    if legend:
        plt.legend(legend)
    plt.grid()
    plt.show()

def extract_pzs(nps, nzs, x, y):
    # TODO
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)

    #print(x)
    #print(y)
    #plot(x, y)

    ma = modal_analysis.ModalAnalysis(rho_threshold=1, N_degree=max(nps, nzs))
    tf = ma.fit_stepresponse(y - y[0], x)
    zs = np.roots(tf['num'])
    ps = np.roots(tf['den'])
    print(zs, ps)

    def pad(xs, n):
        if len(xs) == n:
            return xs
        elif len(xs > n):
            return sorted(xs)[:n]
        else:
            return xs + [float('inf')*(n - len(x))]

    return (pad(ps, nps), pad(zs, nzs))


def dynamic(template):
    # NOTE: the only reason I inherit directly from TemplateMaster
    # here is because I check whether a class is a template by checking
    # whether it's a direct subclass of TemplateMaster
    class Dynamic(template, template_master.TemplateMaster):
        dynamic_reads = {}

        # create function for reading transient in run_single_test
        @classmethod
        def read_transient(self, tester, port, duration):
            r = tester.read(port, style='block', params={'duration':duration})
            self.dynamic_reads[port] = r

        # wrap run_single_test to return read_transient
        @classmethod
        def run_single_test(self, *args, **kwargs):
            ret = super().run_single_test(*args, **kwargs)
            err = ('If you use the Dynamic Template type, you must call '
                'read_transient in your run_single_test!')
            assert len(self.dynamic_reads) > 0, err
            dynamic_reads = self.dynamic_reads
            self.dynamic_reads = {}
            return (ret, dynamic_reads)

        # wrap process_single_test to process the block read
        @classmethod
        def process_single_test(self, reads, *args, **kwargs):
            # we purposely put dynamic_reads in a scope the template creator
            # can access so that they can edit it before we process
            reads_orig, block_reads = reads
            self.dynamic_reads = {p:r.value for p,r in block_reads.items()}
            for p,r in self.dynamic_reads.items():
                print('p')
                if len(r[0]) < 100:
                    print('PROBLEM WITH r')
                    pass
                print(len(r[0]))
            ret_dict = super().process_single_test(reads_orig, *args, **kwargs)
            for port, (x,y) in self.dynamic_reads.items():

                ps, zs = extract_pzs(1, 0, x, y)
                p1 = ps[0]

                # add these ps zs to parameter algebra if they are not already there
                def add_to_p_a(name):
                    if not name in [n for n, _ in self.parameter_algebra]:
                        self.parameter_algebra.append((name, {name:'1'}))
                for n,p in enumerate(ps):
                    name = f'{self.get_name(port)}_p{n}'
                    add_to_p_a(name)
                for n,z in enumerate(zs):
                    name = f'{self.get_name(port)}_z{n}'
                    add_to_p_a(name)


                # add to the dict so they can be used for regression later
                ret_dict[name] = p1
            return ret_dict

    return Dynamic



'''
def plot(xs, ys):
    import matplotlib.pyplot as plt
    plt.plot(xs, ys, '-+')
    plt.grid()
    plt.show()
'''

def debug(test):
    class DebugTest(test):

        def debug(self, tester, port, duration):
            r = tester.get_value(port, params={'style':'block', 'duration': duration})
            self.debug_dict.append((port, r))

        def testbench(self, *args, **kwargs):
            self.debug_dict = []
            self.debug_plot_shown = False
            retval = super().testbench(*args, **kwargs)
            return (self.debug_dict, retval)

        def analysis(self, reads):
            debug_dict, reads_orig = reads

            if not self.debug_plot_shown:
                import matplotlib.pyplot as plt
                leg = []
                bump = 0
                for p, r in debug_dict:
                    s = self.signals.from_spice_pin(p)
                    leg.append(s.spice_name)
                    plt.plot(r.value[0], r.value[1] + bump, '-+')
                    bump += 0.0 # useful for separating clock signals
                plt.grid()
                plt.legend(leg)
                plt.show()
                self.debug_plot_shown = True

            return super().analysis(reads_orig)

    return DebugTest

def make_nondecreasing(ys):
    '''
    Given a list of y values, give a new list of y values that is nondecreasing
    such that the MSE between the two lists is minimized
    '''
    ys = [float(y) for y in ys]
    new_ys = [ys[0]]
    for i in range(1, len(ys)):
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
            for k in range(j_best, i+1):
                new_ys[k] = avg_best

    '''
    import matplotlib.pyplot as plt
    xs = list(range(len(ys)))
    plt.plot(xs, ys, '+')
    plt.plot(xs, new_ys, '--')
    plt.grid()
    plt.show()
    '''

    return new_ys



def invert_function(xs, ys):
    #ys = [float(y) + random.random()*0.02 for y in ys]
    xs = [float(x) for x in xs]
    #xs = list(range(len(xs)))
    #TODO: this is broken for decreasing things
    if ys[0] > ys[-1]:
        temp = [-y for y in ys]
        temp2 = make_nondecreasing(temp)
        ys_up = [-y for y in temp2]
    else:
        ys_up = make_nondecreasing(ys)
    
    new_xs = [xs[0]]
    new_ys = [ys_up[0]]
    float_eps = (ys_up[-1] - ys_up[0])*1e-10 # 1e-10
    for i in range(1, len(xs)-1):
        # look for flat regions
        # start
        if ys_up[i-1] < ys_up[i] - float_eps:
            if ys_up[i] < ys_up[i+1] - float_eps:
                # normal
                new_xs.append(xs[i])
                new_ys.append(ys_up[i])
                #print('normal', i)
            else:
                # start of flat region
                frac = (ys_up[i] - ys[i-1]) / (ys[i] - ys[i-1])
                new_x = xs[i-1] + frac * (xs[i] - xs[i-1])
                new_xs.append(new_x)
                new_ys.append(ys_up[i])
                #print('start of flat', i, frac, xs[i-1:i+2], new_x)
        else:
            if ys_up[i] < ys_up[i+1] - float_eps:
                # end of flat region
                frac = (ys_up[i] - ys[i]) / (ys[i+1] - ys[i])
                new_x = xs[i] + frac * (xs[i+1] - xs[i])
                new_xs.append(new_x)
                new_ys.append(ys_up[i])
                #print('end of flat', i, frac, xs[i-1:i+2], new_x)
            else:
                # middle of flat region - no points necessary
                #print('middle of flat', i)
                pass

    new_xs.append(xs[-1])
    new_ys.append(ys_up[-1])

    #import matplotlib.pyplot as plt
    #plt.plot(xs, ys, '--')
    #plt.plot(xs, ys_up, '+')
    #plt.plot(new_xs, new_ys, '-x')
    #plt.grid()
    #plt.show()

    # TODO I would like to give each flat region a slight tilt because it
    # would help in cases where the true curve is increasing but noise
    # messed it up - we don't want everything in that region collected on
    # one end of the flat region

    endpoints = (new_xs[0], new_xs[-1])
    return interp1d(new_ys, new_xs, bounds_error=False, fill_value=endpoints, assume_sorted=True)


def post_regression_plots(results):
    for param in results.keys():
        reg = results[param]

        y_meas = reg.model.endog
        y_pred = reg.model.predict(reg.params)

        import matplotlib.pyplot as plt
        plt.scatter(y_meas, y_pred)
        plt.title(f'Plot for {param}')
        plt.xlabel('Measured output values')
        plt.ylabel('Predicted output values based on inputs & model')
        # plt.plot([min(y_meas), max(y_meas)], [min(y_meas), max(y_meas)], '--')
        plt.plot([0, max(y_meas)], [0, max(y_meas)], '--')
        plt.grid()
        plt.show()

    return {}