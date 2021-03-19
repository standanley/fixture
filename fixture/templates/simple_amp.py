from fixture import TemplateMaster
from fixture.template_creation_utils import debug

class SimpleAmpTemplate(TemplateMaster):
    required_ports = ['in_single', 'out_single']
    required_info = {
        'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
    }

    #@debug
    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'in_single', 'offset': '1'}
        }

        def input_domain(self):
            # could also use fixture.RealIn(self.in_single.limits, 'my_name')
            return [self.signals.from_template_name('in_single')]

        def testbench(self, tester, values):
            self.debug(tester, self.ports.in_single, 1)
            self.debug(tester, self.ports.out_single, 1)
            in_single = self.ports.in_single
            tester.poke(in_single, values[in_single])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            return tester.get_value(self.ports.out_single)

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

    tests = [Test1]

    

