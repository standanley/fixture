from fixture import TemplateMaster
from fixture import template_creation_utils
from fixture import Real

class SamplerTemplate(TemplateMaster):
    required_ports = ['in_', 'clk', 'out']

    def __init__(self, *args, **kwargs):
        # Some magic constants, maybe pull these from config?
        # NOTE this is before super() because it is used for Test instantiation
        self.nonlinearity_points = 31
        self.aperture_points = 150

        super().__init__(*args, **kwargs)

        # NOTE this must be after super() because it needs ports to be defined
        assert (len(self.ports.out) == len(self.ports.clk),
            'Must have one clock for each output!')


    # @template_creation_utils.debug
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

            # we don't want to create the inverse function here becsue it would
            # only be valid for this particular set of optional inputs
            #template_creation_utils.plot(xs, results)
            #inv = template_creation_utils.invert_function(xs, results)
            #reconstruct_x = [inv(res) for res in results]
            #template_creation_utils.plot(xs, reconstruct_x)
            self.template.temp_inv = template_creation_utils.invert_function(xs, results)

            return ret

    class ApertureTest(TemplateMaster.Test):
        parameter_algebra = {}

        def input_domain(self):
            print('APERATURE TEST INPUT DOMAIN')
            limits = self.ports.in_.limits
            step_start = Real(limits=limits, name='step_start')
            step_end = Real(limits=limits, name='step_end')

            return []

        def testbench(self, tester, values):
            print('APERTURE TEST TESTBENCH')
            settle = float(self.extras['approx_settling_time'])
            wait = 3 * settle

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
            step_start = limits[0]
            step_end = .4#limits[1]

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
            template_creation_utils.plot(xs, ys)

            template_creation_utils.plot(xs, ys_mapped)


            ys_flipped = [-y for y in ys_mapped]
            template_creation_utils.extract_pzs(5, 5, xs, ys_flipped)







    tests = [StaticNonlinearityTest,
             ApertureTest]


