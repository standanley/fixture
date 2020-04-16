from fixture import TemplateMaster
from fixture import template_creation_utils
from fixture import RealIn, BinaryAnalog

class SamplerTemplate(TemplateMaster):
    required_ports = ['in_', 'clk', 'out']

    def __init__(self, *args, **kwargs):
        # Some magic constants, maybe pull these from config?
        # NOTE this is before super() because it is used for Test instantiation
        self.nonlinearity_points = 101

        super().__init__(*args, **kwargs)

        # NOTE this must be after super() because it needs ports to be defined
        assert (len(self.ports.out) == len(self.ports.clk),
            'Must have one clock for each output!')


    @template_creation_utils.debug
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

            return ret

    tests = [StaticNonlinearityTest]


