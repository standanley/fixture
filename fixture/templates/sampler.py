from fixture import TemplateMaster
from fixture import template_creation_utils
from fixture import BinaryAnalogIn, RealIn

import math
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
        self.nonlinearity_points = 3 # 31
        #self.aperture_points = 100# 100

        super().__init__(*args, **kwargs)

        # NOTE this must be after super() because it needs ports to be defined
        assert len(self.ports.out) == len(self.ports.clk), \
            'Must have one clock for each output!'

    def read_value(self, tester, port, wait):
        tester.delay(wait)
        return tester.get_value(port)

    def schedule_clk(self, tester, port, value, wait):
        tester.poke(port, value, delay={'type': 'future', 'wait': wait})

    def interpret_value(self, read):
        return read.value

    @template_creation_utils.debug
    class StaticNonlinearityTest(TemplateMaster.Test):
        num_samples = 1

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
            settle =  float(self.extras['clks']['unit']) * float(self.extras['clks']['period'])
            wait = 2 * settle

            # To see debug plots, also uncomment debug decorator for this class
            np = self.template.nonlinearity_points
            self.debug(tester, self.ports.clk[0], np*wait*2.2)
            self.debug(tester, self.ports.out[0], np*wait*2.2)
            self.debug(tester, self.ports.in_, np*wait*2.2)

            #self.debug(tester, self.template.dut.z_debug, np*wait*2.2)

            self.debug(tester, self.template.dut.clk_v2t_e[0], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2t_e[1], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2t_eb[0], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2t_eb[1], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2t_gated[0], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2t_gated[1], np * wait * 2.2)
            self.debug(tester, self.template.dut.clk_v2t_l[0], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2t_l[1], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2t_lb[0], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2t_lb[1], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2tb_gated[0], np * wait * 2.2)
            #self.debug(tester, self.template.dut.clk_v2tb_gated[1], np * wait * 2.2)

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

                self.template.schedule_clk(tester, self.ports.clk[0], 0, wait)
                tester.delay(wait)
                #tester.poke(self.ports.clk[0], 0)

                read = self.template.read_value(tester, self.ports.out[0], wait)

                # small delay so value is not changed by start of next test
                tester.delay(wait * 0.1)
                results.append((dc, read))
            return results

        def analysis(self, reads):
            results = [self.template.interpret_value(r) for dc, r in reads]
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
            template_creation_utils.plot(temp.y, temp.x)

            return ret

    @template_creation_utils.debug
    class ApertureTest(TemplateMaster.Test):

        # sample_out = should_be_1*value + slope * effective_delay
        # effective_delay = alpha*value + beta*slope + gamma*1

        parameter_algebra = {
            'sample_out': {'should_be_1': 'value',
                           'alpha_times_scale': 'value*slope_over_scale',
                           'beta_times_scale2': 'slope_over_scale**2',
                           'gamma_times_scale': 'slope_over_scale'}
        }
        num_samples = 5

        def input_domain(self):
            limits = self.ports.in_.limits
            settle = 0.5 * float(self.extras['clks']['unit']) * float(self.extras['clks']['period'])
            max_slope = 50 * (limits[1]-limits[0]) / settle

            #step_start = Real(limits=limits, name='step_start')
            #step_end = Real(limits=limits, name='step_end')
            v = RealIn(limits=limits, name='value')
            s = RealIn(limits=(-max_slope, max_slope), name='slope')

            return [v, s]

        def testbench(self, tester, values):
            #settle = float(self.extras['approx_settling_time'])
            settle = float(self.extras['clks']['unit']) * float(self.extras['clks']['period'])
            wait = 1 * settle
            v = values['value']
            s = values['slope']

            # To see debug plots, also uncomment debug decorator for this class
            np = self.template.nonlinearity_points
            debug_length = np * wait * 10
            self.debug(tester, self.ports.clk[0], debug_length)
            self.debug(tester, self.ports.out[0], debug_length)
            self.debug(tester, self.ports.in_, debug_length)

            tester.delay(wait*0.1)

            # feed through to first output, leave the rest off
            # TODO should maybe leave half open, affects charge sharing?
            tester.poke(self.ports.clk[0], 1)
            for p in self.ports.clk[1:]:
                tester.poke(p, 0)

            limits = self.ports.in_.limits
            limit_range = abs(limits[1]-limits[0])
            max_ramp_time = settle/5 # TODO
            if limit_range / abs(s) < max_ramp_time:
                # use the full input range
                start = (limits[0] if s > 0 else limits[1])
                end = (limits[1] if s > 0 else limits[0])

                t_clk = abs((v - start) / s) # TODO abs unnecessary?
                self.template.schedule_clk(tester, self.ports.clk[0], 0, t_clk + wait)

                tester.poke(self.ports.in_, start)
                tester.delay(wait) # because of finite rise time of prev line
                tester.poke(self.ports.in_, 0, delay={
                    'type': 'ramp',
                    'start': start,
                    'stop': end,
                    'rate': s,
                    'etol': limit_range/20
                })

                tester.delay(t_clk)

                # replaced with earlier schedule_clk
                #tester.poke(self.ports.clk[0], 0)

                # The read_value call actually does delay long enough for the thing to settle
                read = self.template.read_value(tester, self.ports.out[0], wait)

                tester.delay(abs(limit_range/s) - t_clk)
                tester.delay(wait * 1)
                print('Using full range for slope', s, ', time', abs(limit_range/s))
                pass

            else:
                # use the full ramp time
                # TODO might be outside range
                def clamp(x):
                    if x < min(limits):
                        return min(limits)
                    elif x > max(limits):
                        return max(limits)
                    else:
                        return x
                ss = clamp(v - s * (max_ramp_time/2))
                se = clamp(v + s * (max_ramp_time/2))


                tester.poke(self.ports.in_, ss)
                tester.delay(wait) # because of finite rise time of prev line
                tester.poke(self.ports.in_, 0, delay={
                    'type': 'ramp',
                    'start': ss,
                    'stop': se,
                    'rate': s,
                    'etol': abs(se-ss)/20
                })

                tester.delay(abs((ss-v)/s))
                tester.poke(self.ports.clk[0], 0)
                read = self.template.read_value(tester, self.ports.out[0], wait)

                tester.delay(max_ramp_time/2)
                print('Using full time for slope', s, ', range', ss, se)

            scale = 10**(int(math.log10(1/settle)))
            slope_over_scale = s / scale
            print('USING SCALE OF', scale)
            return read, slope_over_scale

        def analysis(self, results):
            read, slope_over_scale = results
            v = self.template.interpret_value(read)
            v_mapped = self.template.temp_inv(v)
            return {'sample_out': v_mapped, 'slope_over_scale': slope_over_scale}


        def post_process(self, results):
            vs, ss, samples, ss_scaled = results.values()
            ss_vpns = [s/1e9 for s in ss]

            should_be_1 = 0.996
            alpha = -1.46e-12 * 1e9
            beta = 8.88e-25 * 1e18
            gamma = -2.37e-13 * 1e9

            def f(v, s):
                return sum([
                    should_be_1 * v,
                    alpha * v*s,
                    beta * s**2,
                    gamma * s
                ])

            import numpy as np
            X = np.arange(min(vs), max(vs), 0.01)
            Y = np.arange(min(ss_vpns), max(ss_vpns), (max(ss_vpns) - min(ss_vpns))/100)
            X, Y = np.meshgrid(X, Y)
            #Z = [[f(x, y) for x,y in zip(xx, yy)] for xx, yy in zip(X, Y)]
            Z = f(X, Y)

            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(vs, ss_vpns, samples)
            #ax.plot_surface(X, Y, Z, alpha=0.2)
            ax.set_xlabel('Input Value (V)')
            ax.set_ylabel('Input Slope (V / ns)')
            ax.set_zlabel('Measured (V)')
            plt.show()

            return results
        #def post_process(self, results):
        #    data = self.ys_mapped_data
        #    xs = [x*-1 for x in data[0][2]]
        #    yss = []
        #    yss_aligned = []
        #    legend = []
        #    for ss, se, _, ys in data:
        #        yss.append(ys)
        #        ys_aligned = [(y-ss)/(se-ss) for y in ys]
        #        yss_aligned.append(ys_aligned)
        #        legend.append(f'{ss:.2f} -> {se:.2f}')

        #    template_creation_utils.plot(xs, tuple(yss), legend=legend)
        #    template_creation_utils.plot(xs, tuple(yss_aligned), legend=legend)

        #    def get_delay(xs, ys):
        #        # TODO fencpost error?
        #        cross = (ys[-1] + ys[0]) / 2
        #        i = sum(1 if y < cross else 0 for y in ys)
        #        return xs[i]
        #    voltages_up = []
        #    crosses_up = []
        #    voltages_down = []
        #    crosses_down = []
        #    #dirs = []
        #    for ss, se, _, ys in data:
        #        delay = get_delay(xs, ys)

        #        if ss > se:
        #            crosses_up.append(delay)
        #            voltages_up.append((ss+se)/2)
        #        else:
        #            crosses_down.append(delay)
        #            voltages_down.append((ss+se)/2)


        #    template_creation_utils.plot(voltages_up, crosses_up)
        #    template_creation_utils.plot(voltages_down, crosses_down)
        #    import matplotlib.pyplot as plt
        #    plt.plot(voltages_up, crosses_up, '*')
        #    plt.plot(voltages_down, crosses_down, '+')
        #    plt.legend('Rising edge', 'Falling edge')
        #    plt.grid()
        #    plt.show()





        #    # just pass results through
        #    return results


    tests = [StaticNonlinearityTest,
             ApertureTest]



