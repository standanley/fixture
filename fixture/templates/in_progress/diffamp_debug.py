from fixture import TemplateMaster

class DiffAmpDebugTemplate(TemplateMaster):

    required_ports = ['inp', 'inn', 'outp', 'outn']

    class SweepTest(TemplateMaster.Test):
        parameter_algebra = {
            'gain_meas': {'gain': '1'}
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)',
            'limits_diff': 'Minimum and maximum differential input (V), e.g. (-0.8, 0.8)',
            'limits_cm': 'Minimum and maximum common mode input (V), e.g. (0.4, 0.6)'
        }
        num_samples = 100


        def input_domain(self):
            return []


        def testbench(self, tester, value):
            self.debug(tester, self.ports.inp, 1)
            self.debug(tester, self.ports.inn, 1)
            self.debug(tester, self.ports.outp, 1)
            self.debug(tester, self.ports.outn, 1)
            # self.debug(tester, self.signals.from_spice_name('v_fz').spice_pin, 1)

            # settle from changes to optional inputs
            wait_time = float(self.extras['approx_settling_time']) * 2
            tester.delay(wait_time * 1.0)

            #in_cm, in_diff = value['in_cm'], value['in_diff']
            in_cm = sum(self.extras['limits_cm']) / 2
            in_diff = self.extras['limits_diff'][1] * 0.2
            inp, inn = in_cm + in_diff / 2, in_cm - in_diff / 2
            tester.poke(self.ports.inp, inp)
            tester.poke(self.ports.inn, inn)
            tester.delay(wait_time)

            readp = tester.get_value(self.ports.outp)
            readn = tester.get_value(self.ports.outn)
            return [readp, readn, inp, inn]


        def analysis(self, reads):
            outp = reads[0].value
            outn = reads[1].value
            inp, inn = reads[2:]
            return {'gain_meas': (outp - outn) / (inp - inn)}


        def post_regression(self, results, data):
            return {}
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

    tests_all = [SweepTest]
    tests = tests_all