from fixture import TemplateMaster
from fixture import PlotHelper
import matplotlib.pyplot as plt

class SimpleAmpTemplate(TemplateMaster):
    required_ports = ['in_single', 'out_single']
    required_info = {
        'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
    }

    #@debug
    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'in_single', 'offset': '1'}
        }
        num_samples = 300

        def input_domain(self):
            # could also use fixture.RealIn(self.in_single.limits, 'my_name')
            return [self.signals.from_template_name('in_single')]

        def testbench(self, tester, values):
            self.debug(tester, self.ports.in_single, 1)
            self.debug(tester, self.ports.out_single, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<0>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<1>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<2>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<3>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('vbias').spice_pin, 1)
            in_single = self.ports.in_single
            tester.poke(in_single, values[in_single])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            return tester.get_value(self.ports.out_single)

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

        def post_regression(self, regression_models):
            print('Hello')
            model = regression_models['I(amp_output)'].model
            inputs = model.endog
            outputs = model.exog[:, 0]
            ph = PlotHelper()

            plt.plot(inputs, outputs)

            ph.save_current_plot('amp_test')

            plt.plot(inputs, -outputs + 5)
            ph.save_current_plot('amp_test_2')

            bas = self.signals.binary_analog()
            tas = self.signals.true_analog()

            return {}
    #@debug
    class CubicCompression(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgainc': ('in_single', 'in_single', 'in_single'),
                           'dcgainq': ('in_single', 'in_single'),
                           'dcgain': 'in_single',
                           'offset': '1'}
        }
        num_samples = 300

        def input_domain(self):
            # could also use fixture.RealIn(self.in_single.limits, 'my_name')
            return [self.signals.from_template_name('in_single')]

        def testbench(self, tester, values):
            self.debug(tester, self.ports.in_single, 1)
            self.debug(tester, self.ports.out_single, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<0>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<1>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<2>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<3>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('vbias').spice_pin, 1)
            in_single = self.ports.in_single
            tester.poke(in_single, values[in_single])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            return tester.get_value(self.ports.out_single)

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

        def post_regression(self, results, data):
            return {}
            inputs = data['in_single']
            outputs = data['amp_output']
            ph = PlotHelper()

            plt.plot(inputs, outputs, '+')

            ph.save_current_plot('amp_test')

            plt.plot(inputs, -outputs + 5, 'x')
            ph.save_current_plot('amp_test_2')

            return {}

    tests = [CubicCompression]
    #tests = [Test1]

    

