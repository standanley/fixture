from fixture import TemplateMaster
from fixture.template_creation_utils import dynamic, extract_pzs, remove_repeated_timesteps
from fixture.signals import create_input_domain_signal
import matplotlib.pyplot as plt
from scipy import interpolate


class DifferentialAmpTemplate(TemplateMaster):
    required_ports = ['inp', 'inn', 'outp', 'outn']

    '''
    @dynamic
    class Test2(TemplateMaster.Test):
        parameter_algebra = {
            'out_diff': {'gain':'in_diff', 'gain_from_cm':'in_cm', 'offset':'1'},
            'out_cm': {'gain_to_cm':'in_diff', 'cm_gain':'in_cm', 'cm_offset':'1'},
            'outp': {'A':'inp', 'B':'inn', 'offsetp':'1'},
            'outn': {'C':'inp', 'D':'inn', 'offsetn':'1'}
        }
        num_samples = 100
    '''

    #@debug
    class GainTest(TemplateMaster.Test):
        parameter_algebra = {
            'out_diff': {'gain':'in_diff', 'gain_from_cm':'in_cm', 'offset':'1'},
            'out_cm': {'gain_to_cm':'in_diff', 'cm_gain':'in_cm', 'cm_offset':'1'},
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)',
            'limits_diff': 'Minimum and maximum differential input (V), e.g. (-0.8, 0.8)',
            'limits_cm': 'Minimum and maximum common mode input (V), e.g. (0.4, 0.6)'
        }
        num_samples = 100

        def input_domain(self):
            in_diff = create_input_domain_signal('in_diff', self.extras['limits_diff'])
            in_cm = create_input_domain_signal('in_cm', self.extras['limits_cm'])
            return [in_diff, in_cm]

        def testbench(self, tester, value):
            self.debug(tester, self.ports.inp, 1)
            self.debug(tester, self.ports.inn, 1)
            self.debug(tester, self.ports.outp, 1)
            self.debug(tester, self.ports.outn, 1)
            #self.debug(tester, self.signals.from_spice_name('v_fz').spice_pin, 1)

            # settle from changes to optional inputs
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time * 1.0)

            in_cm, in_diff = value['in_cm'], value['in_diff']
            inp, inn = in_cm + in_diff/2, in_cm - in_diff/2
            tester.poke(self.ports.inp, inp)
            #tester.poke(self.ports.inn, inn,
            #            delay={'type': 'sin', 'freq':2e3, 'amplitude': in_diff, 'offset': in_cm, 'dt': 1/200e3})
            tester.poke(self.ports.inn, inn)
            tester.delay(wait_time)

            readp = tester.get_value(self.ports.outp)
            readn = tester.get_value(self.ports.outn)
            return [readp, readn, inp, inn]


        def analysis(self, reads):
            outp = reads[0].value
            outn = reads[1].value
            return {'out_diff': outp - outn, 'out_cm': (outp + outn) / 2,
                    'outp': outp, 'outn': outn, 'inp': reads[2], 'inn': reads[3]}

        def post_regression(self, results, data):
            if hasattr(self, 'IS_DEBUG_MODE'):
                # TODO this does not work with the newer post_regression
                # signature, but rather than fix it we should do it in an
                # automated way with plot_helper
                for param in results.keys():
                    reg = results[param]

                    y_meas = reg.model.endog
                    y_pred = reg.model.predict(reg.params)

                    plt.scatter(y_meas, y_pred)
                    plt.title(f'Plot for {param}')
                    plt.xlabel('Measured output values')
                    plt.ylabel('Predicted output values based on inputs & model')
                    plt.plot([0, max(y_meas)], [0, max(y_meas)], '--')
                    plt.grid()
                    plt.show()

            return {}


    #@debug
    class DynamicTest(TemplateMaster.Test):
        parameter_algebra = {
            #'p1': {'cm_to_p1': 'in_cm', 'const_p1': '1'},
            #'p2': {'cm_to_p2': 'in_cm', 'const_p2': '1'},
            #'z1': {'cm_to_z1': 'in_cm', 'const_z1': '1'},
            'p1': {'const_p1': '1'},
            'p2': {'const_p2': '1'},
            'z1': {'const_z1': '1'},
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
        }
        num_samples = 50

        def input_domain(self):
            in_cm = create_input_domain_signal('in_cm', self.extras['limits_cm'])
            vmin, vmax = self.extras['limits_diff']
            step_size = create_input_domain_signal('step_size', ((vmax-vmin)/2, vmax-vmin))
            step_pos = create_input_domain_signal('step_pos', (0, 1))
            return [in_cm, step_size, step_pos]

        def testbench(self, tester, value):
            wait_time = float(self.extras['approx_settling_time'])*2
            self.debug(tester, self.ports.inp, 1)
            self.debug(tester, self.ports.inn, 1)
            self.debug(tester, self.ports.outp, 1)
            self.debug(tester, self.ports.outn, 1)
            #self.debug(tester, self.signals.from_spice_name('v_fz').spice_pin, 1)

            # settle from changes to optional inputs
            tester.delay(wait_time * 1.0)

            in_cm = value['in_cm']
            vmin, vmax = self.extras['limits_diff']
            step_start = vmin + ((vmax-vmin) - value['step_size']) * value['step_pos']
            step_end = step_start + value['step_size']
            inp_start, inn_start = in_cm + step_start/2, in_cm - step_start/2
            inp_end, inn_end = in_cm + step_end/2, in_cm - step_end/2

            tester.poke(self.ports.inp, inp_start)
            tester.poke(self.ports.inn, inn_start)

            # let everything settle before the step
            tester.delay(wait_time)

            tester.poke(self.ports.inp, inp_end)
            tester.poke(self.ports.inn, inn_end)
            rp = tester.get_value(self.ports.outp, params=
                    {'style':'block', 'duration': wait_time}
                )
            rn = tester.get_value(self.ports.outn, params=
                    {'style':'block', 'duration': wait_time}
                )

            # we wait a tiny bit extra so we're not messed up by the next edge
            tester.delay(wait_time * 1.1)

            return [rp, rn]


        def analysis(self, reads):
            outp = reads[0].value
            outn = reads[1].value

            # haven't written good logic for if the timesteps don't match
            if len(outp[0]) != len(outn[0]) or any(outp[0] != outn[0]):
                outp = remove_repeated_timesteps(*outp)
                outn = remove_repeated_timesteps(*outn)
                # hmm, timesteps don't match
                # reasmple n with p's timesteps? Not ideal, but good enough
                interpn = interpolate.InterpolatedUnivariateSpline(outn[0], outn[1])
                resampled_outn = interpn(outp[0])
                outn = outp[0], resampled_outn

            # we want to cut some off, but leave at least 60-15*2 ??
            CUTOFF = 0#min(max(0, len(outp[0]) - 60), 15)

            step_start_output = outp[1][0] - outn[1][0]
            outdiff = outp[0], outp[1] - outn[1] - step_start_output

            # FLIP
            #outdiff = outdiff[0], -1 * outdiff[1]


            ps, zs = extract_pzs(2, 1, outdiff[0][CUTOFF:], outdiff[1][CUTOFF:])
            list(ps).sort(key=abs)
            zs.sort()


            return {'p1': ps[0], 'p2': ps[1], 'z1': zs[0]}

        def post_regression(self, results, data):
            #return {}
            if hasattr(self, 'IS_DEBUG_MODE'):
                for param in results.keys():
                    reg = results[param]

                    y_meas = reg.model.endog
                    y_pred = reg.model.predict(reg.params)

                    plt.scatter(y_meas, y_pred)
                    plt.title(f'Plot for {param}')
                    plt.xlabel('Measured output values')
                    plt.ylabel('Predicted output values based on inputs & model')
                    #plt.plot([min(y_meas), max(y_meas)], [min(y_meas), max(y_meas)], '--')
                    plt.plot([0, max(y_meas)], [0, max(y_meas)], '--')
                    plt.grid()
                    plt.show()

            return {}

    class BodeTest(TemplateMaster.Test):
        parameter_algebra = {
            'p1': {'cm_to_p1': 'in_cm', 'const_p1': '1'},
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
        }
        num_samples = 1

        def input_domain(self):
            in_cm = create_input_domain_signal('in_cm', self.extras['limits_cm'])
            return [in_cm]

        def testbench(self, tester, value):
            import numpy as np

            wait_time = float(self.extras['approx_settling_time'])*2
            cm = value['in_cm']

            self.debug(tester, self.ports.inp, 1)
            self.debug(tester, self.ports.inn, 1)
            self.debug(tester, self.ports.outp, 1)
            self.debug(tester, self.ports.outn, 1)

            in_cm = value['in_cm']
            vmin, vmax = self.extras['limits_diff']
            # use 1/2 range

            amp = min(abs(vmin), abs(vmax)) / 2
            fnom_log10 = int(np.log10(1/float(self.extras['approx_settling_time']))+.5)
            freqs = np.logspace(fnom_log10-1, fnom_log10+5, 13)

            reads = {}
            for freq in freqs:
                tester.poke(self.ports.inp, 0, delay={
                    'type': 'sin',
                    'freq': freq,
                    'amplitude': amp,
                    'offset': cm,
                    'dt': 1 / (freq * 1000)
                })
                tester.poke(self.ports.inn, 0, delay={
                    'type': 'sin',
                    'freq': freq,
                    'amplitude': -amp,
                    'offset': cm,
                    'dt': 1 / (freq * 1000)
                })

                period5 = 5/freq
                print('JUST SET PERIOD5', period5)
                readp = tester.get_value(self.ports.outp, params={
                    'style': 'block',
                    'duration': period5
                })
                readn = tester.get_value(self.ports.outn, params={
                    'style': 'block',
                    'duration': period5
                })
                tester.delay(period5)

                # if the super-fast clock gets left running at the end it's bad
                tester.poke(self.ports.inn, 0)
                tester.poke(self.ports.inp, 0)

                reads[freq] = (readp, readn)

            return reads


        def analysis(self, reads):
            freqs = []
            amps = []
            for f, (outp, outn) in reads.items():
                # haven't written logic for if the timesteps don't match
                outp, outn = outp.value, outn.value
                assert all(outp[0] == outn[0])
                outdiff = outp[0], outp[1] - outn[1]
                MARGIN = 5
                amp = (max(outdiff[1][MARGIN:-MARGIN])
                       - min(outdiff[1][MARGIN:-MARGIN]))
                print('GOT AMP', amp)
                print(max(outdiff[1][MARGIN:-MARGIN]),
                       min(outdiff[1][MARGIN:-MARGIN]))
                freqs.append(f)
                amps.append(amp)

            if hasattr(self, 'IS_DEBUG_MODE'):
                plt.loglog(freqs, amps, '-+')
                plt.grid()
                plt.show()


            return {'p1': ps[0], 'p2': ps[1], 'z1': zs[0]}


    tests = [GainTest, DynamicTest]
    #tests = [DynamicTest]

