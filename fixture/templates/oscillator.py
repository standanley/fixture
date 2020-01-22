from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput

class OscillatorTemplate(TemplateMaster):
    required_ports = ['out']
    parameter_algebra = {
            'frequency': 'const'
        }

    @classmethod
    def specify_test_inputs(self):
        return []

    @classmethod
    def run_single_test(self, tester, values):
        approx_period = 1 / float(self.extras['approx_frequency'])
        tester.delay(approx_period * 20)
        return tester.read(self.out, style='frequency')


    @classmethod
    def process_single_test(self, read):
        results = {'frequency': read.value}
        return results


    

