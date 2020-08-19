from fixture import TemplateMaster
from fixture import template_creation_utils
from fixture import BinaryAnalogIn, RealIn

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import trapz



def characterize_aperture_simple(step_response):
    # assume you only get the value and slope at t=0
    # what's the optimal reconstruction?
    # we just find a scale factor alpha=integral(A(t)*t)
    # The reconstruction is then alpha*slope+value, assuming integral(A(t))=1

    ts_orig, hs = step_response

    # convert step response to impulse
    dts = np.diff(ts_orig)
    dhs = np.diff(hs)
    ys = dhs / dts
    ts = ts_orig[:-1]
    xs = [t*y for t,y in zip(ts, ys)]
    alpha = trapz(ys, x=xs)
    return [alpha]

def fit_step_response(step_response, target_area):
    '''
    TODO: this function assumes uniform time steps
    Given a measured aperture step response and known height
    of the step (target_area for the impulse response), calculate
    two best-fit triangles for the left and right halves.
    t0 is the time of the peak of the triangle.
    The left half of the triangle has width w and height h, the
    right half has w2 and h2, we guarantee the combined area of
    the triangles is target_area
    '''
    ts_orig, hs = step_response

    # convert step response to impulse
    dts = np.diff(ts_orig)
    dhs = np.diff(hs)
    ys = dhs / dts
    ts = ts_orig[:-1]

    # Initially guess the max point of the impulse response
    # is the triangle peak, with 90% area to the left
    t0_guess = ts[np.argmax(ys)]
    h_guess = max(ys)
    w_guess = target_area * .9 / (.5 * h_guess)
    h2_guess = h_guess

    # minimize function doesn't like when parameters
    # are on wildly different scales, so we normalize
    # by this unit
    unit = 1 / t0_guess

    x0 = np.array([t0_guess * unit, w_guess * unit, h_guess / unit, h2_guess / unit])

    def piecewise(t, x):
        '''
        Given a time t and set of parameters x, find the height
        of the triangle at time t (or 0 if outside the triangle)
        '''
        (t0, w, h, h2) = x
        t0, w, h, h2 = t0 / unit, w / unit, h * unit, h2 * unit
        area = .5 * w * h
        area2 = target_area - area
        w2 = area2 / (.5 * h2)
        if t < t0 - w:
            return 0
        elif t < t0:
            return h * (t - (t0 - w)) / w
        elif t < t0 + w2:
            return h2 * ((t0 + w2) - t) / w2
        else:
            return 0

    def fun(x, ts, ys):
        '''
        Function to be passed to scipy.optimize.minimize
        Determines total square error of triangle fit
        '''
        # retval = sum((y-piecewise(t, x))**2 for t, y in zip(ts, ys))
        s = 0
        for t, y in zip(ts, ys):
            a = (y - piecewise(t, x))
            b = a ** 2
            s += b
        retval = s
        return retval

    result = minimize(fun, x0, args=(ts, ys))
    x_min = result.x

    # Plot results
    ys_triangle = [piecewise(t, x_min) for t in ts]
    legend = ['Measured', 'Fit']
    ts_negative = [-1*t for t in ts]
    #template_creation_utils.plot(ts_negative, (ys, ys_triangle), legend)

    # convert from normalized optimizer units to regular untis
    (t0, w, h, h2) = x_min
    result = (t0 / unit, w / unit, h * unit, h2 * unit)
    return result

class SamplerTemplate(TemplateMaster):
    required_ports = ['in_', 'clk', 'out']

    def __init__(self, *args, **kwargs):
        # Some magic constants, maybe pull these from config?
        # NOTE this is before super() because it is used for Test instantiation
        self.nonlinearity_points = 10# 31
        self.aperture_points = 100# 100

        super().__init__(*args, **kwargs)

        # NOTE this must be after super() because it needs ports to be defined
        assert (len(self.ports.out) == len(self.ports.clk),
            'Must have one clock for each output!')


    #@template_creation_utils.debug
    class StaticNonlinearityTest(TemplateMaster.Test):
        def __init__(self, *args, **kwargs):
            # set parameter algebra before parent checks it
            nl_points = args[0].nonlinearity_points
            self.parameter_algebra = {}
            for i in range(nl_points):
                self.parameter_algebra[f'nl_{i}'] = {f'nonlinearity_{i}': '1'}

            super().__init__(*args, **kwargs)

        def input_domain(self):
            return []

        def testbench(self, tester, values):
            wait = 5 * float(self.extras['approx_settling_time'])

            # To see debug plots, also uncomment debug decorator for this class
            #np = self.template.nonlinearity_points
            #self.debug(tester, self.ports.clk[0], np*wait*2.2)
            #self.debug(tester, self.ports.out[0], np*wait*2.2)
            #self.debug(tester, self.ports.in_, np*wait*2.2)

            tester.delay(wait*0.2)

            # feed through to first output, leave the rest off
            # TODO should maybe leave half open, affects charge sharing?
            tester.poke(self.ports.clk[0], 1)
            for p in self.ports.clk[1:]:
                tester.poke(p, 0)

            limits = self.ports.in_.limits
            num = self.template.nonlinearity_points
            results = []
            for i in range(num):
                dc = limits[0] + i * (limits[1] - limits[0]) / (num-1)
                tester.poke(self.ports.clk[0], 1)
                tester.poke(self.ports.in_, dc)

                tester.delay(wait)
                tester.poke(self.ports.clk[0], 0)
                tester.delay(wait)
                read = tester.get_value(self.ports.out[0])
                # small delay so value is not changed by start of next test
                tester.delay(wait * 0.1)
                results.append((dc, read))

            return results

        def analysis(self, reads):
            results = [r.value for dc, r in reads]
            xs = [dc for dc, r in reads]

            ret = {f'nl_{i}': v for i, v in enumerate(results)}

            # we don't want to create the inverse function here because it would
            # only be valid for this particular set of optional inputs
            #template_creation_utils.plot(xs, results)
            #inv = template_creation_utils.invert_function(xs, results)
            #reconstruct_x = [inv(res) for res in results]
            #template_creation_utils.plot(xs, reconstruct_x)
            #template_creation_utils.plot(xs, results)
            self.template.temp_inv = template_creation_utils.invert_function(xs, results)

            temp = self.template.temp_inv
            #template_creation_utils.plot(temp.y, temp.x)

            return ret

    @template_creation_utils.debug
    class ApertureTest(TemplateMaster.Test):
        parameter_algebra = {
            #'t0': {'aperture_delay': '1'},
            #'w': {'aperture_w_left': '1'},
            #'h': {'aperture_h_left': '1'},
            # 'w2': {'aperture_w_right': '1'}, determined by area constraint
            #'h2': {'aperture_h_right': '1'}
            'aperture_alpha': {'alpha_const':'1', 'alpha_dir_adjust': 'step_dir'}
        }

        def input_domain(self):
            limits = self.ports.in_.limits
            #step_start = Real(limits=limits, name='step_start')
            #step_end = Real(limits=limits, name='step_end')
            step_dir = BinaryAnalogIn(name='step_dir')
            step_pos = RealIn(limits=(0,1), name='step_pos')

            return [step_dir, step_pos]

        def testbench(self, tester, values):
            settle = float(self.extras['approx_settling_time'])
            wait = 3 * settle

            # To see debug plots, also uncomment debug decorator for this class
            np = self.template.nonlinearity_points
            self.debug(tester, self.ports.clk[0], np*wait*2.2)
            self.debug(tester, self.ports.out[0], np*wait*2.2)
            self.debug(tester, self.ports.in_, np*wait*2.2)

            tester.delay(wait*0.2)

            # feed through to first output, leave the rest off
            # TODO should maybe leave half open, affects charge sharing?
            tester.poke(self.ports.clk[0], 1)
            for p in self.ports.clk[1:]:
                tester.poke(p, 0)

            limits = self.ports.in_.limits
            step_start = 0.1#limits[0]
            step_end = .7#limits[1]
            limit_range = limits[1]-limits[0]
            step_size = limit_range/4
            step_start = limits[0] + (limit_range - step_size) * values['step_pos']
            step_end = step_start + step_size
            if values['step_dir']:
                step_start, step_end = step_end, step_start
            else:
                pass

            num = self.template.aperture_points
            t_min = -0.5 * settle
            t_max = 0.25 * settle
            results = []
            for i in range(num):
                t = t_min + i * (t_max - t_min) / (num-1)
                tester.poke(self.ports.clk[0], 1)
                tester.poke(self.ports.in_, step_start)

                # the clock always happens at the same exact time,
                # and we move the step earlier (t_min) to later (t_max)

                if t < 0:
                    # step first
                    tester.delay(wait + t)
                    tester.poke(self.ports.in_, step_end)
                    tester.delay(-t)
                    tester.poke(self.ports.clk[0], 0)
                    tester.delay(wait)
                else:
                    # clock first
                    tester.delay(wait)
                    tester.poke(self.ports.clk[0], 0)
                    tester.delay(t)
                    tester.poke(self.ports.in_, step_end)
                    tester.delay(wait - t)

                read = tester.get_value(self.ports.out[0])
                results.append((t, read))

            return ((step_start, step_end), results)

        def analysis(self, results):
            (step_start, step_end), reads = results
            xs = [float(x) for x,gv in reads]
            ys = [float(gv.value) for x,gv in reads]

            ys_mapped = [self.template.temp_inv(y) for y in ys]
            data = (step_start, step_end, xs, ys_mapped)
            if hasattr(self, 'ys_mapped_data'):
                self.ys_mapped_data.append(data)
            else:
                self.ys_mapped_data = [data]

            # In this plot the left side is what we see of steps that happen
            # early compared to the clock, and right side is steps that
            # happen late, so the plot shows a falling edge.
            # It's not time invariant with respect to a later step, it's sorta
            # reverse time invariant - that means if we want the impulse
            # response represented in the same way we should take derivative
            # and then negate it.
            #template_creation_utils.plot(xs, ys)
            #template_creation_utils.plot(xs, ys_mapped)
            xs_neg = [-1*x for x in xs]
            print('About to do original/mapped plot')
            #template_creation_utils.plot(xs_neg, (ys, ys_mapped), ['Original', 'Mapped'])
            print('JUST DID PLOTS')


            ys_flipped = [-y for y in ys_mapped]
            #template_creation_utils.extract_pzs(5, 5, xs, ys_flipped)
            step_size = ys_flipped[-1]-ys_flipped[0]
            #t0, w, h, h2 = fit_step_response((xs, ys_flipped), step_size)
            #return {'t0': t0,
            #        'w': w,
            #        'h': h,
            #        'h2': h2}
            (alpha,) = characterize_aperture_simple((xs, ys_flipped))
            return {'aperture_alpha': alpha}

        def post_process(self, results):
            data = self.ys_mapped_data
            xs = [x*-1 for x in data[0][2]]
            yss = []
            yss_aligned = []
            legend = []
            for ss, se, _, ys in data:
                yss.append(ys)
                ys_aligned = [(y-ss)/(se-ss) for y in ys]
                yss_aligned.append(ys_aligned)
                legend.append(f'{ss:.2f} -> {se:.2f}')

            template_creation_utils.plot(xs, tuple(yss), legend=legend)
            template_creation_utils.plot(xs, tuple(yss_aligned), legend=legend)

            def get_delay(xs, ys):
                # TODO fencpost error?
                cross = (ys[-1] + ys[0]) / 2
                i = sum(1 if y < cross else 0 for y in ys)
                return xs[i]
            voltages_up = []
            crosses_up = []
            voltages_down = []
            crosses_down = []
            #dirs = []
            for ss, se, _, ys in data:
                delay = get_delay(xs, ys)

                if ss > se:
                    crosses_up.append(delay)
                    voltages_up.append((ss+se)/2)
                else:
                    crosses_down.append(delay)
                    voltages_down.append((ss+se)/2)


            template_creation_utils.plot(voltages_up, crosses_up)
            template_creation_utils.plot(voltages_down, crosses_down)
            import matplotlib.pyplot as plt
            plt.plot(voltages_up, crosses_up, '*')
            plt.plot(voltages_down, crosses_down, '+')
            plt.legend('Rising edge', 'Falling edge')
            plt.grid()
            plt.show()





            # just pass results through
            return results


    tests = [StaticNonlinearityTest,
             ApertureTest]



