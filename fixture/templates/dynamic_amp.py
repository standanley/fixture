from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput
from fixture import template_creation_utils

@template_creation_utils.dynamic
class DynamicAmpTemplate(TemplateMaster):
    __name__ = 'abc123'
    required_ports = ['in_single', 'out_single']
    parameter_algebra = [
        ('amp_output', {'gain':'in_single', 'offset':'1'})
    ]

    @classmethod
    def specify_test_inputs(self):
        # could also use fixture.RealIn(self.in_single.limits, 'my_name')
        return [self.in_single]

    @classmethod
    def specify_test_outputs(self):
        return [TestVectorOutput('amp_output')]

    @classmethod
    def run_single_test(self, tester, values):
        wait_time = float(self.extras['approx_settling_time'])*2
        self.read_transient(tester, self.out_single, wait_time*1)
        tester.poke(self.in_single, values['in_single'])
        tester.delay(wait_time)
        tester.expect(self.out_single, 0, save_for_later=True)
        return tester.read(self.out_single)


    @classmethod
    def process_single_test(self, read_out_single):
        results = {'amp_output': read_out_single.value}
        return results

