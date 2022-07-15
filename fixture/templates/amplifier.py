from fixture import TemplateMaster
from fixture import PlotHelper
import matplotlib.pyplot as plt
import numpy as np
from fixture.signals import create_input_domain_signal, SignalArray
from fixture.template_creation_utils import extract_pzs


class AmplifierTemplate(TemplateMaster):
    required_ports = ['input', 'output']
    required_info = {
        'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
    }

    class DCTest(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'input',
                           'offset': '1'},
            #output_diff = dcgain_from_diff*input_diff + ... + offset
            #output_cm = dcgain_from_diff*input_diff + ... + offset
            #'amp_output_copy': {'just_offset': '1'}
            #'amp_output_copy': {'dcgain_3': 'input**3',
            #               'dcgain_2': 'input**2',
            #               'dcgain_1': 'input',
            #               'dcgain_0': '1'}
        }
        num_samples = 300

        def input_domain(self):
            # could also use fixture.RealIn(self.input.limits, 'my_name')
            return [self.signals.from_template_name('input')]

        def testbench(self, tester, values):
            debug_time = 500e-9
            #self.debug(tester, self.signals.input, debug_time)
            self.debug(tester, self.signals.output[0], debug_time)
            self.debug(tester, self.signals.input[0], debug_time)
            #self.debug(tester, self.signals.input[0], debug_time)
            #self.debug(tester, self.signals.input[1], debug_time)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<0>').spice_pin, debug_time)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<1>').spice_pin, debug_time)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<2>').spice_pin, debug_time)
            #self.debug(tester, self.signals.from_spice_name('cm_adj<3>').spice_pin, debug_time)
            #self.debug(tester, self.signals.from_spice_name('vbias').spice_pin, debug_time)
            input = self.signals.input
            tester.poke(input, values[input])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            # next line shouldn't be necessary
            tester.poke(input, values[input])
            return tester.get_value(self.signals.output)

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

        def post_regression(self, regression_models, regression_dataframe):
            return {}
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


    class AbsoluteValue(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'input',
                           'abs_param': 'input_abs',
                           'offset': '1'}
        }
        num_samples = 300
        input_vector_mapping = {'input_abs': 'input'}
        out_vector = ['amp_output']

        def input_domain(self):
            # could also use fixture.RealIn(self.input.limits, 'my_name')
            return [self.signals.from_template_name('input')]

        def testbench(self, tester, values):
            self.debug(tester, self.signals.input, 1)
            self.debug(tester, self.signals.output, 1)
            input = self.signals.input
            tester.poke(input, values[input])
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            out = tester.get_value(self.signals.output)
            return (out, values[input])

        def analysis(self, reads):
            output, input = reads
            results = {'amp_output': output.value,
                       'input_abs': np.abs(input)}
            return results

    class DynamicTFTest(TemplateMaster.Test):
        NP = 2
        NZ = 1
        parameter_algebra = {
            **{f'numerator{i}': {f'const_numerator{i}': '1'} for i in range(1, NZ+1)},
            **{f'denominator{i}': {f'const_denominator{i}': '1'} for i in range(1, NP+1)},
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
        }
        num_samples = 100

        def input_domain(self):
            max_step = 0.5
            input = self.signals.from_template_name('input')
            if isinstance(input, SignalArray):
                # vectored in
                values = [x.value for x in input]
                step_sizes = [((v[1]-v[0])*max_step, v[1]-v[0]) for v in values]
                size = [create_input_domain_signal(f'step_size[{i}]',
                                                   step_size)
                        for i, step_size in enumerate(step_sizes)]
                pos = [create_input_domain_signal(f'step_pos[{i}]', (0, 1))
                       for i in range(len(step_sizes))]
                return size + pos
            else:
                vmin, vmax = input.value
                step_size = create_input_domain_signal('step_size',
                                                       ((vmax-vmin)*max_step, vmax-vmin))
                step_pos = create_input_domain_signal('step_pos', (0, 1))
                return [step_size, step_pos]

        def testbench(self, tester, values):
            wait_time = float(self.extras['approx_settling_time'])*2
            #self.debug(tester, self.ports.inp, 1)
            #self.debug(tester, self.ports.inn, 1)
            #self.debug(tester, self.ports.outp, 1)
            #self.debug(tester, self.ports.outn, 1)
            #self.debug(tester, self.signals.from_spice_name('v_fz').spice_pin, 1)
            self.debug(tester, self.signals.input, 1)
            self.debug(tester, self.signals.output, 1)

            # settle from changes to optional inputs
            #tester.delay(wait_time * 1.0)


            if isinstance(self.signals.input, SignalArray):
                ranges = [s.value for s in self.signals.input]
                vmin = np.array([r[0] for r in ranges])
                vmax = np.array([r[1] for r in ranges])
            else:
                vmin, vmax = self.signals.input.value
            step_start = vmin + ((vmax-vmin) - values['step_size']) * values['step_pos']
            step_end = step_start + values['step_size']

            tester.poke(self.signals.input, step_start)

            # let everything settle before the step
            tester.delay(wait_time*2)

            tester.poke(self.signals.input, step_end)
            read = tester.get_value(self.signals.output, params=
            {'style':'block', 'duration': wait_time}
                                    )

            # we wait a tiny bit extra so we're not messed up by the next edge
            tester.delay(wait_time * 1.1)

            return [read]


        def analysis(self, reads):
            out = reads[0].value
            t = out[0]
            v = out[1] - out[1][0]

            ps, zs = extract_pzs(self.NP, self.NZ, t, v)

            den_full = np.poly(ps)
            num_full = np.poly(zs)

            den = np.real(den_full[1:])
            num = np.real(num_full[1:])

            poles = {f'denominator{i+1}': c for i, c in enumerate(den)}
            zeros = {f'numerator{i+1}': c for i, c in enumerate(num)}
            return {**poles, **zeros}

        def post_regression(self, regression_models, regression_dataframe):
            print()
            return {}



    class DynamicTest(TemplateMaster.Test):
        NP = 2
        NZ = 1
        parameter_algebra = {
            **{f'p{i}': {f'const_p{i}': '1'} for i in range(1, NP+1)},
            **{f'z{i}': {f'const_z{i}': '1'} for i in range(1, NZ+1)},
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
        }
        num_samples = 50

        def input_domain(self):
            max_step = 0.5
            input = self.signals.from_template_name('input')
            if isinstance(input, SignalArray):
                # vectored in
                values = [x.value for x in input]
                step_sizes = [((v[1]-v[0])*max_step, v[1]-v[0]) for v in values]
                size = [create_input_domain_signal(f'step_size[{i}]',
                                                   step_size)
                        for i, step_size in enumerate(step_sizes)]
                pos = [create_input_domain_signal(f'step_pos[{i}]', (0, 1))
                       for i in range(len(step_sizes))]
                return size + pos
            else:
                vmin, vmax = input.value
                step_size = create_input_domain_signal('step_size',
                                                       ((vmax-vmin)*max_step, vmax-vmin))
                step_pos = create_input_domain_signal('step_pos', (0, 1))
                return [step_size, step_pos]

        def testbench(self, tester, values):
            wait_time = float(self.extras['approx_settling_time'])*2
            #self.debug(tester, self.ports.inp, 1)
            #self.debug(tester, self.ports.inn, 1)
            #self.debug(tester, self.ports.outp, 1)
            #self.debug(tester, self.ports.outn, 1)
            #self.debug(tester, self.signals.from_spice_name('v_fz').spice_pin, 1)
            self.debug(tester, self.signals.input, 1)
            self.debug(tester, self.signals.output, 1)

            # settle from changes to optional inputs
            #tester.delay(wait_time * 1.0)


            if isinstance(self.signals.input, SignalArray):
                ranges = [s.value for s in self.signals.input]
                vmin = np.array([r[0] for r in ranges])
                vmax = np.array([r[1] for r in ranges])
            else:
                vmin, vmax = self.signals.input.value
            step_start = vmin + ((vmax-vmin) - values['step_size']) * values['step_pos']
            step_end = step_start + values['step_size']

            tester.poke(self.signals.input, step_start)

            # let everything settle before the step
            tester.delay(wait_time*2)

            tester.poke(self.signals.input, step_end)
            read = tester.get_value(self.signals.output, params=
                {'style':'block', 'duration': wait_time}
                                  )

            # we wait a tiny bit extra so we're not messed up by the next edge
            tester.delay(wait_time * 1.1)

            return [read]


        def analysis(self, reads):
            out = reads[0].value
            t = out[0]
            v = out[1] - out[1][0]

            ps, zs = extract_pzs(self.NP, self.NZ, t, v)
            list(ps).sort(key=abs)
            zs.sort()

            poles = {f'p{i+1}': p for i, p in enumerate(ps)}
            zeros = {f'z{i+1}': z for i, z in enumerate(zs)}
            return {**poles, **zeros}

        def post_regression(self, regression_models, regression_dataframe):
            print()
            return {}


    tests = [
        DCTest,
        DynamicTFTest,
        #CubicCompression,
        #AbsoluteValue,
    ]

    

