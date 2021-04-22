from fixture import TemplateMaster
from fixture import template_creation_utils

class OscillatorTemplate(TemplateMaster):
    required_ports = ['out']
    required_info = {
        'approx_frequency': 'Ballpark guess for the output frequency (Hz)'
    }

    #@template_creation_utils.debug
    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'frequency': 'const'
        }

        def input_domain(self):
            return []

        def testbench(self, tester, values):
            #self.debug(tester, self.ports.out, 1)
            #self.debug(tester, self.signals.from_spice_name('adj').spice_pin, 1)
            approx_period = 1 / float(self.extras['approx_frequency'])
            tester.delay(approx_period * 5)
            res = tester.get_value(self.ports.out, params={'style':'frequency'})
            #res = tester.get_value(self.ports.out)
            return res

        @classmethod
        def analysis(self, read):
            results = {'frequency': read.value}
            return results

    tests = [Test1]

