from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput

class SimpleAmpTemplate(TemplateMaster):
    required_ports = ['in_single', 'out_single']

    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'in_single', 'offset': '1'}
        }

        def input_domain(self):
            # could also use fixture.RealIn(self.in_single.limits, 'my_name')
            return [self.ports.in_single]

        def testbench(self, tester, values):
            tester.poke(self.ports.in_single, values['in_single'])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            return tester.get_value(self.ports.out_single)

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

    tests = [Test1]

    

