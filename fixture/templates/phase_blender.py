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

        # always between 0 and 1
        #rand_phase_offset = values[1]
        # "random" value within the specified range
        #phase_offset = offset_range[0] + rand_phase_offset*(offset_range[1]-offset_range[0])

        phase_diff = values['in_phase_diff']
        print('phase diff', phase_diff, '\tsel', values['sel'])

        #tester.poke(self.in_a, 1)
        #tester.delay(phase_diff / freq)
        #tester.poke(self.in_a, 1)
        #tester.delay((0.5-phase_diff) / freq)
        #tester.poke(self.in_a, 0)
        #tester.delay(phase_diff / freq)
        #tester.poke(self.in_a, 0)
        #tester.delay((0.5-phase_diff) / freq)


        tester.poke(self.in_a, 0, delay={
            'freq': freq,
            })
        tester.delay(phase_diff / freq)
        tester.poke(self.in_b, 0, delay={
            'freq': freq,
            })

        #utils.poke_binary_analog(tester, self.sel, values['sel'])

        # TODO get this next line to work
        #tester.poke(self.sel, values['sel'])
        for i in range(len(self.sel)):
            tester.poke(self.sel[i], values['sel[%d]'%i])

        # wait 5 cycles for things to settle
        tester.delay(5 / freq)

        tester.expect(self.out, 0, save_for_later=True)

        # these are just to force a wave dump on these nodes
        tester.expect(self.in_a, 0, save_for_later=True)
        tester.expect(self.in_b, 0, save_for_later=True)
        tester.expect(self.sel[0], 0, save_for_later=True)
        tester.expect(self.sel[1], 0, save_for_later=True)
        tester.expect(self.sel[2], 0, save_for_later=True)

        out_phase = tester.read(self.out, style='phase', params={
            'ref': self.in_a
            })
        return [out_phase]

    @classmethod
    def process_single_test(self, reads):
        out_phase = reads[0].value
        #results = []
        #results.append(tester.results_raw[tester.result_counter])
        #tester.result_counter += 1
        ret = {'out_phase': out_phase}
        return ret


