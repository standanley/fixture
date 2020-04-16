from fixture import TemplateMaster
from fixture import template_creation_utils
from fixture import RealIn, BinaryAnalog

class SamplerTemplate(TemplateMaster):
    required_ports = ['in_', 'clk', 'out']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (len(self.ports.out) == len(self.ports.clk),
            'Must have one clock for each output!')

        # Some magic constants, maybe pull these from config?
        self.nonlinearity_points = 101

    @template_creation_utils.debug
    class StaticNonlinearityTest(TemplateMaster.Test):
        parameter_algebra = {
            #'out_phase': {'gain':'in_phase_diff*sel', 'offset':'1'}
        }

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

            #template_creation_utils.plot(xs, results)
            template_creation_utils.invert_function(xs, results)

            return 'not implemented'

    tests = [StaticNonlinearityTest]


