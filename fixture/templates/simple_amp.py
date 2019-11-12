#from .. import templates
#from ..templates import *
from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput

class SimpleAmpTemplate(TemplateMaster):
    __name__ = 'abc123'
    required_ports = ['in_single', 'out_single']
    parameter_algebra = ['amp_output ~ gain:amp_input + offset']

    @classmethod
    def specify_test_inputs(self):
        input_limits = self.in_single.limits
        in_ = TestVectorInput(input_limits, 'amp_input')
        return [in_]

    @classmethod
    def specify_test_outputs(self):
        return [TestVectorOutput('amp_output')]

    @classmethod
    def run_single_test(self, tester, value):
        tester.poke(self.in_single, value[0])
        wait_time = float(self.extras['approx_settling_time'])*2
        tester.delay(wait_time)
        tester.expect(self.out_single, 0, save_for_later=True)

    @classmethod
    def process_single_test(self, tester):
        results = []
        results.append(tester.results_raw[tester.result_counter])
        tester.result_counter += 1
        # for an amp, for now, no post-processing is required
        return results


    

