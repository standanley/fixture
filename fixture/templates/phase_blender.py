from fixture import TemplateMaster
from fixture import RealIn
from fixture.template_creation_utils import debug
#from fixture.real_types import BinaryAnalogKind, TestVectorOutput

class PhaseBlenderTemplate(TemplateMaster):
    required_ports = ['in_a', 'in_b', 'out']
    required_info = {
        'phase_offset_range': 'phase offset between in_a and in_b, must be between 0 and 1',
        'frequency': 'Input clock frequency (Hz)'
    }

    #@debug
    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'out_delay': {'gain':'in_phase_delay', 'offset':'1'}
        }
        num_samples = 100


        def input_domain(self):
            # offset range is in units of "periods", so 0.5 means in_a and in_b are in quadrature
            offset_range = self.extras.get('phase_offset_range', (0, .5))
            freq = float(self.extras['frequency'])
            offset_delay_range = tuple((offset / freq for offset in offset_range))
            diff = RealIn(offset_delay_range)
            # TODO make this part of the instantiation of RealIn
            diff.name = 'in_phase_delay'

            # could make a new test vector with same params as sel, or just use sel itself
            # new_sel = Array(len(self.sel), BinaryAnalog)
            return [diff]

        def testbench(self, tester, values):
            freq = float(self.extras['frequency'])

            # always between 0 and 1
            #rand_phase_offset = values[1]
            # "random" value within the specified range
            #phase_offset = offset_range[0] + rand_phase_offset*(offset_range[1]-offset_range[0])

            self.debug(tester, self.ports.in_a, 1/freq*100)
            self.debug(tester, self.ports.in_b, 1/freq*100)
            self.debug(tester, self.ports.out, 1/freq*100)
            #self.debug(tester, self.template.dut.thm_sel_bld[0], 1/freq*100)
            #self.debug(tester, self.template.dut.sel[0], 1/freq*100)


            in_phase_delay = values['in_phase_delay']

            tester.poke(self.ports.in_a, 0, delay={
                'freq': freq,
                })
            tester.delay(in_phase_delay)
            tester.poke(self.ports.in_b, 0, delay={
                'freq': freq,
                })

            # TODO get this next line to work
            #tester.poke(self.ports.sel, values['sel'])
            #for i in range(len(self.ports.sel)):
            #    tester.poke(self.ports.sel[i], values[self.ports.sel[i]])

            # wait 5 cycles for things to settle
            tester.delay(5 / freq)

            # TODO: what was this next line for?
            #tester.expect(self.ports.out, 0, save_for_later=True)

            # these are just to force a wave dump on these nodes
            # tester.read(self.ports.in_a)
            # tester.read(self.ports.in_b)
            # tester.read(self.ports.sel[0])
            # tester.read(self.ports.sel[1])
            # tester.read(self.ports.sel[2])
            #tester.expect(self.ports.in_a, 0, save_for_later=True)
            #tester.expect(self.ports.in_b, 0, save_for_later=True)
            #tester.expect(self.ports.sel[0], 0, save_for_later=True)
            #tester.expect(self.ports.sel[1], 0, save_for_later=True)
            #tester.expect(self.ports.sel[2], 0, save_for_later=True)


            out_phase = tester.get_value(self.ports.out, params={
                'style': 'phase',
                'ref': self.ports.in_a
                })
            #out_phase = None

            # wait a touch longer because I had issues when the simulation ended exactly as the measurement was taken
            tester.delay(2/freq)
            tester.poke(self.ports.in_a, 0)
            tester.poke(self.ports.in_b, 0)
            tester.delay(1/freq)
            return [out_phase]

        def analysis(self, reads):
            freq = float(self.extras['frequency'])
            out_phase = reads[0].value
            out_delay = out_phase / freq

            # fix for unintentional wrapping below 0
            period = 1 / freq
            if out_delay > 0.9 * period:
                out_delay -= period

            ret = {'out_delay': out_delay}
            return ret

        def post_process(self, results):
            # if the delay is close to 1 period we may have some measurements
            # wrap around to the beginning of the next period. This tries to
            # deal with that.
            print(results)
            period = 1/float(self.extras['frequency'])
            # all the measured out_delays are in results['out_delay']
            # we don't know where in the period they are clumped but we
            # will assume they are all in one clump
            # First find the gap between i and i+1
            outs = results['out_delay']
            N = len(outs)
            gaps = []
            outs_sorted = sorted(outs)
            for i in range(N):
                gap = outs_sorted[(i+1)%N] - outs_sorted[i]
                if gap < 0:
                    gap += period
                gaps.append(gap)
            biggest_gap = gaps.index(max(gaps))

            # put our cut point at the end of the gap,
            # anything smaller in value than the cut point should
            # be moved forward one period
            cut = outs_sorted[(biggest_gap+1)%N]
            for i in range(N):
                if outs[i] < cut:
                    outs[i] += period

            # TODO check this. also, probably should either edit in place or return, not both
            results['out_delay'] = outs

            ## avoid precision issues in regression
            ## TODO fix this next line
            #in_phase_delay = self.inputs_analog[0]
            #for k in ['out_delay', in_phase_delay]:
            #    results[k] = [x*1e6 for x in results[k]]
            return results

    tests = [Test1]


