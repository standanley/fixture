from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput
import fixture.template_creation_utils as utils

class PhaseBlenderTemplate(TemplateMaster):
    __name__ = 'phase_blender_template'
    required_ports = ['in_a', 'in_b', 'sel', 'out']
    parameter_algebra = ['I(out_phase - in_a_phase) ~ gain:I((in_b_phase-in_a_phase)*in_b) + offset']

    @classmethod
    def specify_test_inputs(self):
        a = TestVectorInput(self.in_a.limits, 'in_a_phase')
        b = TestVectorInput(self.in_b.limits, 'in_b_phase')
        sel = TestVectorInput((0, 1), 'sel_fraction')
        return [a, b, sel]

    @classmethod
    def specify_test_outputs(self):
        return [TestVectorOutput('out_phase')]

    @classmethod
    def run_single_test(self, tester, values):
        freq = self.extras['frequency']
        offset_range = self.extras.get('phase_offset_range', (0, .5))

        # always between 0 and 1
        rand_phase_offset = values[1]
        # "random" value within the specified range
        phase_offset = offset_range[0] + rand_phase_offset*(offset_range[1]-offset_range[0])

        tester.poke(self.in_a, 0, delay={
            'freq': freq,
            'phase': values[0]
            })
        tester.poke(self.in_b, 0, delay={
            'freq': freq,
            'phase': values[0] + phase_offset
            })
        utils.poke_binary_analog(tester, self.sel, values[2])

        # wait 5 cycles for things to settle
        tester.delay(5 / freq)

        tester.expect(self.out_single, 0, save_for_later=True)

    @classmethod
    def process_single_test(self, tester):
        results = []
        results.append(tester.results_raw[tester.result_counter])
        tester.result_counter += 1
        # for an amp, for now, no post-processing is required
        return results


    

