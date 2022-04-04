from fixture import TemplateMaster
from fixture import PlotHelper
import matplotlib.pyplot as plt
import numpy as np

class AmplifierTemplate(TemplateMaster):
    required_ports = ['input', 'output']
    required_info = {
        'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
    }

    #@debug
    class DCTest(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'input', 'gainsq': 'input^2', 'offset': '1'}
        }
        num_samples = 300

        def input_domain(self):
            # could also use fixture.RealIn(self.input.limits, 'my_name')
            return [self.signals.from_template_name('input')]

        def testbench(self, tester, values):
            self.debug(tester, self.signals.input, 1)
            self.debug(tester, self.signals.output, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<0>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<1>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<2>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<3>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('vbias').spice_pin, 1)
            input = self.signals.input
            tester.poke(input, values[input])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            return tester.get_value(self.signals.output)

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

        def post_regression(self, regression_models, regression_dataframe):
            return{}
            print('Hello')
            model = regression_models['amp_output_vec[0]'].model
            y = model.endog
            xs = model.exog

            #in_diff = xs[:,1]
            #in_cm = xs[:,2]
            in_diff = xs[:,2]
            in_cm = xs[:,4]

            ph = PlotHelper()

            plt.plot(in_diff, y, '*')

            ph.save_current_plot('amp_test')

            pred_tmp = 0.011 + 2.156*in_diff - 0.026*in_cm
            plt.plot([-2, 2], [-2, 2], '--')
            plt.plot(y, pred_tmp,'*')
            ph.save_current_plot('amp_test_2')


            from scipy.interpolate import griddata
            gridx, gridy = np.mgrid[-.7:.7:100j, 0.4:0.5:100j]
            #points = xs[:, 1:3]
            points = np.vstack((in_diff, in_cm)).T
            test = griddata(points, y, (gridx, gridy), method='linear')

            cp = plt.contourf(gridx, gridy, test)
            plt.colorbar(cp)
            plt.show()



            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(in_diff, in_cm, y)
            #ax.scatter(in_diff, in_cm, pred_tmp)
            plt.show()
            #bas = self.signals.binary_analog()
            #tas = self.signals.true_analog()



            return {}
    #@debug
    class CubicCompression(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgainc': 'input**3',
                           'dcgainq': 'input**2',
                           'dcgain': 'input',
                           'offset': '1'}
        }
        num_samples = 300

        def input_domain(self):
            # could also use fixture.RealIn(self.input.limits, 'my_name')
            return [self.signals.from_template_name('input')]

        def testbench(self, tester, values):
            self.debug(tester, self.signals.input, 1)
            self.debug(tester, self.signals.output, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<0>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<1>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<2>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<3>').spice_pin, 1)
            #self.debug(tester, self.signals.from_spice_name('vbias').spice_pin, 1)
            input = self.signals.input
            tester.poke(input, values[input])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            return tester.get_value(self.signals.output)

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

        def post_regression(self, results, data):
            return {}
            inputs = data['input']
            outputs = data['amp_output']
            ph = PlotHelper()

            plt.plot(inputs, outputs, '+')

            ph.save_current_plot('amp_test')

            plt.plot(inputs, -outputs + 5, 'x')
            ph.save_current_plot('amp_test_2')

            return {}

    tests = [
        DCTest,
        #CubicCompression,
    ]

    

