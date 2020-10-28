#from fixture import TemplateMaster
from fixture.create_testbench2 import EmptyTemplate

class SimpleAmpTemplate(EmptyTemplate):
    required_ports = ['in_single', 'out_single']

    class Test1(EmptyTemplate.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'in_single', 'offset': '1'}
        }

        def set_signals(self):
            # could also use fixture.RealIn(self.in_single.limits, 'my_name')
            in_ = self.in_single
            in_.get_random = True

        def testbench(self, tester, values):
            tester.poke(self.spice.in_single, values['in_single'])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            return tester.get_value(self.spice.out_single)

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

    tests = [Test1]

    

