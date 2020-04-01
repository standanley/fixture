from fixture import TemplateMaster
from fixture import RealIn, BinaryAnalog
#import fixture.template_creation_utils as utils
#from fixture.real_types import BinaryAnalogKind, TestVectorOutput

class PhaseBlenderTemplate(TemplateMaster):
    required_ports = ['in_a', 'in_b', 'sel', 'out']
    parameter_algebra = {
        'out_phase': {'gain':'in_phase_diff*sel', 'offset':'1'}
    }

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
    def run_single_test(self, tester, values):
        freq = self.extras['frequency']

        # always between 0 and 1
        #rand_phase_offset = values[1]
        # "random" value within the specified range
        #phase_offset = offset_range[0] + rand_phase_offset*(offset_range[1]-offset_range[0])

        phase_diff = values['in_phase_diff']

        tester.poke(self.in_a, 0, delay={
            'freq': freq,
            })
        tester.delay(phase_diff / freq)
        tester.poke(self.in_b, 0, delay={
            'freq': freq,
            })

        # TODO get this next line to work
        #tester.poke(self.sel, values['sel'])
        for i in range(len(self.sel)):
            tester.poke(self.sel[i], values[self.sel[i]])

        # wait 5 cycles for things to settle
        tester.delay(5 / freq)

        # TODO: what was this next line for?
        #tester.expect(self.out, 0, save_for_later=True)

        # these are just to force a wave dump on these nodes
        # tester.read(self.in_a)
        # tester.read(self.in_b)
        # tester.read(self.sel[0])
        # tester.read(self.sel[1])
        # tester.read(self.sel[2])
        #tester.expect(self.in_a, 0, save_for_later=True)
        #tester.expect(self.in_b, 0, save_for_later=True)
        #tester.expect(self.sel[0], 0, save_for_later=True)
        #tester.expect(self.sel[1], 0, save_for_later=True)
        #tester.expect(self.sel[2], 0, save_for_later=True)

        out_phase = tester.get_value(self.out, params={
            'style': 'phase',
            'ref': self.in_a
            })

        # wait a touch longer because I had issues when the simulation ended exactly as the measurement was taken
        tester.delay(2/freq)
        tester.poke(self.in_a, 0)
        tester.poke(self.in_b, 0)
        tester.delay(1/freq)
        return [out_phase]

    @classmethod
    def process_single_test(self, reads):
        out_phase = reads[0].value
        ret = {'out_phase': out_phase}
        return ret


