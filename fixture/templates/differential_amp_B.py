from fixture import TemplateMaster
from fixture.template_creation_utils import dynamic, debug, extract_pzs
from fixture.signals import create_input_domain_signal


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
    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'out_diff': {'gain':'in_diff', 'gain_from_cm':'in_cm', 'offset':'1'},
            'out_cm': {'gain_to_cm':'in_diff', 'cm_gain':'in_cm', 'cm_offset':'1'},
            'outp': {'A':'inp', 'B':'inn', 'offsetp':'1'},
            'outn': {'C':'inp', 'D':'inn', 'offsetn':'1'}
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
        }
        num_samples = 100

        def input_domain(self):
            in_diff = create_input_domain_signal('in_diff', self.extras['limits_diff'])
            in_cm = create_input_domain_signal('in_cm', self.extras['limits_cm'])
            return [in_diff, in_cm]

        def testbench(self, tester, value):
            self.debug(tester, self.ports.inp, 1)
            self.debug(tester, self.ports.inn, 1)
            in_cm, in_diff = value['in_cm'], value['in_diff']
            inp, inn = in_cm + in_diff/2, in_cm - in_diff/2
            tester.poke(self.ports.inp, inp)
            #tester.poke(self.ports.inn, inn,
            #            delay={'type': 'sin', 'freq':2e3, 'amplitude': in_diff, 'offset': in_cm, 'dt': 1/200e3})
            tester.poke(self.ports.inn, inn)
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)

            readp = tester.get_value(self.ports.outp)
            readn = tester.get_value(self.ports.outn)
            return [readp, readn, inp, inn]


        def analysis(self, reads):
            outp = reads[0].value
            outn = reads[1].value
            return {'out_diff': outp - outn, 'out_cm': (outp + outn) / 2,
                    'outp': outp, 'outn': outn, 'inp': reads[2], 'inn': reads[3]}

        def post_regression(self, results):
            diff_out = results['I(out_diff)']

            y_meas = diff_out.model.endog
            y_pred = diff_out.model.predict(diff_out.params)

            import matplotlib.pyplot as plt
            plt.scatter(y_meas, y_pred)
            plt.xlabel('Measured output values')
            plt.ylabel('Predicted output values based on inputs & model')
            plt.plot([min(y_meas), max(y_meas)], [min(y_meas), max(y_meas)], '--')
            plt.grid()
            plt.show()

            return {}


    @debug
    class DynamicTest(TemplateMaster.Test):
        parameter_algebra = {
            'p1': {'cm_to_p1': 'in_cm', 'const_p1': '1'},
            'p2': {'cm_to_p2': 'in_cm', 'const_p2': '1'},
            'z1': {'cm_to_z1': 'in_cm', 'const_z1': '1'},
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
        }
        num_samples = 10

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

            # haven't written logic if the timeteps don't match
            assert all(outp[0] == outn[0])

            CUTOFF = 20

            outdiff = outp[0], outp[1] - outn[1]

            ps, zs = extract_pzs(2, 1, outdiff[0][CUTOFF:], outdiff[1][CUTOFF:])


            return {'p1': ps[0], 'p2': ps[1], 'z1': zs[0]}

        def post_regression(self, results):
            for param in results.keys():
                reg = results[param]

                y_meas = reg.model.endog
                y_pred = reg.model.predict(reg.params)

                import matplotlib.pyplot as plt
                plt.scatter(y_meas, y_pred)
                plt.title(f'Plot for {param}')
                plt.xlabel('Measured output values')
                plt.ylabel('Predicted output values based on inputs & model')
                plt.plot([min(y_meas), max(y_meas)], [min(y_meas), max(y_meas)], '--')
                plt.grid()
                plt.show()

            return {}

    tests = [DynamicTest]

