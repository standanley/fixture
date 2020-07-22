from fixture import TemplateMaster
from fixture import template_creation_utils
from fixture import Real

import numpy as np
from scipy.optimize import minimize

class SamplerTemplate(TemplateMaster):
    required_ports = ['in_', 'clk', 'out']

    def __init__(self, *args, **kwargs):
        # Some magic constants, maybe pull these from config?
        # NOTE this is before super() because it is used for Test instantiation
        self.nonlinearity_points = 31
        self.aperture_points = 100

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
            template_creation_utils.plot(xs, results)
            self.template.temp_inv = template_creation_utils.invert_function(xs, results)

            temp = self.template.temp_inv
            template_creation_utils.plot(temp.y, temp.x)

            return ret

    @template_creation_utils.debug
    class ApertureTest(TemplateMaster.Test):
        parameter_algebra = {
            't0': {'aperture_delay': '1'},
            'w': {'aperture_w_left': '1'},
            'h': {'aperture_h_left': '1'},
            # 'w2': {'aperture_w_right': '1'}, determined by area constraint
            'h2': {'aperture_h_right': '1'}
        }

        def input_domain(self):
            print('APERTURE TEST INPUT DOMAIN')
            limits = self.ports.in_.limits
            step_start = Real(limits=limits, name='step_start')
            step_end = Real(limits=limits, name='step_end')

            return []

        def testbench(self, tester, values):
            print('APERTURE TEST TESTBENCH')
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

            return results

        def fit_step_response(self, step_response, target_area):
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
            template_creation_utils.plot(ts, (ys, ys_triangle), legend)

            # convert from normalized optimizer units to regular untis
            (t0, w, h, h2) = x_min
            result = (t0 / unit, w / unit, h * unit, h2 * unit)
            return result

        def analysis(self, reads):
            xs = [float(x) for x,gv in reads]
            ys = [float(gv.value) for x,gv in reads]

            ys_mapped = [self.template.temp_inv(y) for y in ys]

            # In this plot the left side is what we see of steps that happen
            # early compared to the clock, and right side is steps that
            # happen late, so the plot shows a falling edge.
            # It's not time invariant with respect to a later step, it's sorta
            # reverse time invariant - that means if we want the impulse
            # response represented in the same way we should take derivative
            # and then negate it.
            #template_creation_utils.plot(xs, ys)
            #template_creation_utils.plot(xs, ys_mapped)
            template_creation_utils.plot(xs, (ys, ys_mapped), ['Original', 'Mapped'])
            print('JUST DID PLOTS')


            ys_flipped = [-y for y in ys_mapped]
            #template_creation_utils.extract_pzs(5, 5, xs, ys_flipped)
            step_size = ys_flipped[-1]-ys_flipped[0]
            t0, w, h, h2 = self.fit_step_response((xs, ys_flipped), step_size)
            return {'t0': t0,
                    'w': w,
                    'h': h,
                    'h2': h2}


    tests = [StaticNonlinearityTest,
             ApertureTest]


