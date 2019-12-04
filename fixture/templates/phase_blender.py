from fixture import TemplateMaster
from fixture import RealIn, BinaryAnalog
import fixture.template_creation_utils as utils
from fixture.real_types import BinaryAnalogKind, TestVectorOutput

class PhaseBlenderTemplate(TemplateMaster):
    __name__ = 'phase_blender_template'
    required_ports = ['in_a', 'in_b', 'sel', 'out']
    #parameter_algebra = ['I(out_phase - in_a_phase) ~ gain:I((in_b_phase-in_a_phase)*sel) + offset']
    parameter_algebra = ['out_phase ~ gain:I(in_phase_diff*sel) + offset']

    @classmethod
    def specify_test_inputs(self):
        offset_range = self.extras.get('phase_offset_range', (0, .5))
        diff = RealIn(offset_range)
        # TODO make this part of the instantiationo f RealIn
        diff.name = 'in_phase_diff'

        # could make a new test vector with same params as sel, or just use sel itself
        # new_sel = Array(len(self.sel), BinaryAnalog)
        return [diff, self.sel]

    @classmethod
    def specify_test_outputs(self):
        return [TestVectorOutput('out_phase')]

    @classmethod
    def run_single_test(self, tester, values):
        freq = self.extras['frequency']
        print('got freq', freq, 'of type', type(freq))

        # always between 0 and 1
        #rand_phase_offset = values[1]
        # "random" value within the specified range
        #phase_offset = offset_range[0] + rand_phase_offset*(offset_range[1]-offset_range[0])

        phase_a = 0
        phase_diff = values['in_phase_diff']

        tester.poke(self.in_a, 0, delay={
            'freq': freq,
            'phase': phase_a
            })
        tester.poke(self.in_b, 0, delay={
            'freq': freq,
            'phase': phase_a + phase_diff
            })

        #utils.poke_binary_analog(tester, self.sel, values['sel'])
        tester.poke(self.sel, values['sel'])

        # wait 5 cycles for things to settle
        tester.delay(5 / freq)

        tester.expect(self.out, 0, save_for_later=True)

    @classmethod
    def process_single_test(self, tester):
        results = []
        results.append(tester.results_raw[tester.result_counter])
        tester.result_counter += 1
        # for an amp, for now, no post-processing is required
        return results


    

