import ast
import functools

import numpy as np
import scipy

from fixture import TemplateMaster, PlotHelper
from fixture import template_creation_utils
from fixture import signals
import fixture
from fixture.signals import SignalOut

Regression = fixture.regression.Regression
import math
import matplotlib.pyplot as plt
from fault import domain_read
from fixture import ChannelUtil


class SamplerTemplate(TemplateMaster):
    required_ports = ['in_', 'clk', 'out']
    required_info = {
        #'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)',
        #'max_slope': 'Approximate maximum slope of the input signal to be sampled (V/s)'
        'clks': 'Timing information for one period including clock edges and sampling and read times'
    }

    def __init__(self, *args, **kwargs):
        # Some magic constants, maybe pull these from config?
        # NOTE this is before super() because it is used for Test instantiation
        self.nonlinearity_points = 11 # 31

        # we have to do this before getting input domains, which happens
        # in the call to super
        extras = args[3]
        signal_manager = args[2]

        # NOTE I think this block has to be before super() because that's when
        # the individual tests get copies of these signals ??
        # but must come after for self.signals to be defined?
        if 'clks' in extras:
            settle = float(extras['clks']['unit']) * float(extras['clks']['period'])
            if 'approx_settling_time' not in extras:
                extras['approx_settling_time'] = settle
            extras['cycle_time'] = settle

            clks = extras['clks']
            #clks = {k: v for k, v in clks.items() if (k != 'unit' and k != 'period')}
            for clk, v in clks.items():
                if type(v) == dict and 'max_jitter' in v:
                    x = v['max_jitter']
                    signal_manager.add(signals.SignalIn(
                        (-x, x),
                        'analog',
                        True,
                        False,
                        None,
                        None,
                        clk+'_jitter',
                        True
                    ))


        super().__init__(*args, **kwargs)


        # NOTE this must be after super() because it needs ports to be defined

        #if not hasattr(self.ports.clk, '__getitem__'):
        #    #self.ports.mapping['clk'] = [self.ports.mapping['clk']]
        #    #self.ports.mapping['out'] = [self.ports.mapping['out']]
        #    pass
        #else:
        #    assert len(self.ports.out) == len(self.ports.clk), \
        #        'Must have one clock for each output!'


    def read_value(self, tester, port, wait):
        tester.delay(wait)
        s = self.signals.from_circuit_pin(port)
        return tester.get_value(s)

    def interpret_value(self, read):
        return read.value

    def get_clock_offset_domain(self):
        return []

    def schedule_clk(self, tester, output, num_periods=1, pos=0.5, jitters={}):
        # play num_periods periods, each time with "output" coming after
        # period_len*pos
        assert 'clks' in self.extras

        clks = self.extras['clks']
        if hasattr(self.signals, 'ignore'):
            for s_ignore in self.signals.ignore:
                if s_ignore is None:
                    continue
                if s_ignore.spice_name not in clks:
                    tester.poke(s_ignore.spice_pin, 0)

        if isinstance(self.signals.clk, list):
            clk_list = self.signals.clk
            num_samplers = len(clk_list)
            #desired_sampler = clk_list.index(main)
        else:
            #main = self.get_name_circuit(port)
            num_samplers = 1
            #desired_sampler = 0
        unit = float(clks['unit'])
        period = clks['period']
        #clks = {self.signals.from_circuit_name(k): v for k,v in clks.items()
        #        if (k!='unit' and k!='period')}
        clks_new = {}
        outs = {}
        for k, v in clks.items():
            try:
                if isinstance(k, SignalOut):
                    x = k
                else:
                    x = self.signals.from_circuit_name(k)
                if x in self.signals.clk:
                    clks_new[x] = v
                elif x in self.signals.out:
                    outs[x] = v
                elif x in self.signals.ignore:
                    # if it's in clks, we should poke it
                    clks_new[x] = v
                else:
                    assert False, f'clk dict port {x} not clk or out or ignore'
            except KeyError:
                continue
        clks = clks_new


        # main_period_start_time used to be rising edge of main clock
        # we were trying to put the falling edge in the middle of the period
        # Now we want to sampling time in the middle of the period
        #assert main in clks, f'"clks" spec missing {main}'
        #main_period_start_time = [t for t,v in clks[main].items() if v==1][0]
        assert output in outs
        main_dict_rev = {v: k for k, v in outs[output].items()}
        sample_time = main_dict_rev['sample']
        time_sample_to_read = (main_dict_rev['read'] - main_dict_rev['sample']) * unit
        time_sample_to_read %= (period * unit)

        # shift the user-given period such that main_period_start_time is position
        # rising edge just happened

        for i in range(num_samplers):
            #offset = ((i - desired_sampler + num_samplers) % num_samplers) / num_samplers
            #period_start_time = (main_period_start_time + period * offset) % period
            # pick start time s.t. X plays after waiting pos*period
            period_start_time = (sample_time - pos * period + period) % period

            clks_transform = {}

            for clk in clks:
                temp = []
                name = clk[0].spice_name if isinstance(clk, list) else clk.spice_name
                jitter_name = name + '_jitter'
                jitter = jitters.get(jitter_name, 0)
                for p in range(num_periods):
                    for time, val in clks[clk].items():
                        if time == 'max_jitter':
                            # not actually a time spec,
                            # should maybe be removed from dict earleir
                            continue
                        time_transform = time + (period - period_start_time)
                        # TODO is there a reason we didn't mod here?
                        if time_transform > period:
                            time_transform -= period
                        time_transform_shift = time_transform + p * period
                        temp.append((time_transform_shift + jitter, val))
                clks_transform[clk] = sorted(temp)

            # shift that one period s.t. the falling edge of the main clk
            # happens after exactly "wait"
            # simply ignore any edges that would've been in the past
            # shift is the time in seconds from now until the period start
            shift = 0# wait - period_start_time * unit
            if shift < 0:
                print('Cannot run a full period when scheduling clk edges', i)

            for clk, edges in clks_transform.items():
                t = 0
                waits = [0]
                values = [0 if edges[0][1] else 1]
                for time, value in edges:
                    x = time * unit + shift
                    if x < 0:
                        print('Skipping edge', value, 'for', clk, i)
                        continue
                    waits.append(x - t)
                    t = x
                    values.append(value)
                #current_clk_bus = getattr(self.dut, clk)
                if not hasattr(clk, '__getitem__'):
                    current_clk = clk
                else:
                    current_clk = clk[i]
                assert waits[0] >= 0 and all(w > 0 for w in waits[1:])
                tester.poke(current_clk.spice_pin, 0, delay={
                    'type': 'future',
                    'waits': waits,
                    'values': values
                })

        # amount of time between sampling and looking at the sampled voltage
        return time_sample_to_read


    class StaticNonlinearityTest(TemplateMaster.Test):
        num_samples = 3 #10

        def __init__(self, *args, **kwargs):
            print("STATIC INIT")
            # set parameter algebra before parent checks it
            nl_points = args[0].nonlinearity_points
            self.parameter_algebra = {}
            for i in range(nl_points):
                self.parameter_algebra[f'nl_{i}'] = {f'nonlinearity_{i}': '1'}

            super().__init__(*args, **kwargs)

        def input_domain(self):
            return self.template.get_clock_offset_domain()

        def testbench(self, tester, values):
            #print('Chose jitter value', values['clk_v2t_l_jitter'], 'In static test')
            period = float(self.extras['cycle_time'])
            clk = self.signals.clk[0] if hasattr(self.signals.clk, '__getitem__') else self.signals.clk
            assert isinstance(clk, signals.SignalIn)

            # To see debug plots, also uncomment debug decorator for this class
            np = self.template.nonlinearity_points
            debug_time = np * period * 22
            self.debug(tester, clk.spice_pin, debug_time)
            #for p in self.ports.out:
            #    self.debug(tester, p, debug_time)
            self.debug(tester, self.signals.from_circuit_name('out<0>').spice_pin, debug_time)
            self.debug(tester, self.ports.in_, debug_time)
            if hasattr(self.ports, 'debug'):
                self.debug(tester, self.ports.debug, debug_time)

            #self.debug(tester, self.template.dut.z_debug, debug_time)

            #self.debug(tester, self.template.dut.clk_v2t_e[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_e[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_eb[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_eb[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_gated[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_gated[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_l[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_l[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_lb[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2t_lb[1], debug_time)
            #self.debug(tester, self.template.dut.clk_v2tb_gated[0], debug_time)
            #self.debug(tester, self.template.dut.clk_v2tb_gated[1], debug_time)

            # prevents overlapping points from confusing fault
            tester.delay(period*0.01)

            # feed through to first output, leave the rest off
            # TODO should maybe leave half open, affects charge sharing?
            tester.poke(clk.spice_pin, 1)
            if hasattr(self.ports.clk, '__getitem__'):
                for p in self.ports.clk[1:]:
                    tester.poke(p, 0)

            # get output port
            if hasattr(self.ports.out, '__getitem__'):
                p = self.ports.out[0]
            else:
                p = self.ports.out

            limits = self.signals.in_.value
            num = self.template.nonlinearity_points
            results = []
            prev_val = 0
            for i in range(num):
                dc = limits[0] + i * (limits[1] - limits[0]) / (num-1)
                #tester.poke(clk.spice_pin, 1)
                tester.poke(self.ports.in_, prev_val)
                tester.delay(0.01 * period)
                tester.poke(self.ports.in_, dc)
                prev_val = dc

                settle_time = self.template.schedule_clk(tester, self.signals.out[0], 1, 0.5, values)
                tester.delay(period / 2)
                print('1: delaying', period/2)

                # delays time "wait" for things to settle before reading
                tester.delay(settle_time)
                print('2: delaying ', settle_time)
                read = self.template.read_value(tester, p, 0)

                if settle_time < period / 2:
                    tester.delay(period / 2 - settle_time)
                    print('3: delaying', period/2 - settle_time)
                results.append((dc, read))
            tester.poke(self.ports.in_, prev_val)
            tester.delay(0.01*period)


            tester.delay(2*period)

            return results

        def analysis(self, reads):
            results = [self.template.interpret_value(r) for dc, r in reads]
            xs = [dc for dc, r in reads]

            ret = {f'nl_{i}': v for i, v in enumerate(results)}

            # we don't want to create the inverse function here because it would
            # only be valid for this particular set of optional inputs
            #template_creation_utils.plot(xs, results)
            #inv = template_creation_utils.invert_function(xs, results)
            #reconstruct_x = [inv(res) for res in results]
            #template_creation_utils.plot(xs, reconstruct_x)
            #template_creation_utils.plot(xs, results)
            self.template.temp_inv = template_creation_utils.invert_function(xs, results)

            #temp = self.template.temp_inv
            #template_creation_utils.plot(temp.y, temp.x)

            return ret

        def post_regression(self, regression_models, regression_dataframe):
            return {}
            limits = self.signals.in_.value
            num = self.template.nonlinearity_points
            xs = []
            for i in range(num):
                dc = limits[0] + i * (limits[1] - limits[0]) / (num-1)
                xs.append(dc)
            ys = []
            for i in range(num):
                val = regression_models[f'nonlinearity_{i}']['1']
                ys.append(val)

            plt.figure()
            plt.plot(xs, ys, '*')
            plt.grid()
            plt.show()
            return {}

    class ApertureTest(TemplateMaster.Test):

        # sample_out = should_be_1*value + slope * effective_delay
        # effective_delay = alpha*value + beta*slope + gamma*1

        parameter_algebra = {
            'sample_out': {'should_be_1': 'value',
                           'alpha_times_scale': 'value*slope_over_scale',
                           'beta_times_scale2': 'slope_over_scale**2',
                           'gamma_times_scale': 'slope_over_scale'}
        }
        num_samples = 20

        def input_domain(self):
            limits = self.signals.from_template_name('in_').value
            max_slope = 2*self.extras['max_slope'] # 50 * (limits[1]-limits[0]) / settle

            v = signals.create_input_domain_signal('value', limits)
            s = signals.create_input_domain_signal('slope', (-max_slope, max_slope))
            jitter_domain = self.template.get_clock_offset_domain()

            return [v, s] + jitter_domain

        def testbench(self, tester, values):
            #print('Chose jitter value', values['clk_v2t_l_jitter'], 'in dynamic test')
            period = float(self.extras['cycle_time'])
            v = values['value']
            s = values['slope']
            clk = self.signals.clk[0] if hasattr(self.signals.clk, '__getitem__') else self.signals.clk
            assert isinstance(clk, signals.SignalIn)

            # To see debug plots, also uncomment debug decorator for this class
            np = self.template.nonlinearity_points
            debug_length = np * period * 30
            self.debug(tester, clk.spice_pin, debug_length)
            for p in self.ports.out:
                self.debug(tester, p, debug_length)
            self.debug(tester, self.ports.in_, debug_length)
            #self.debug(tester, self.template.dut.clk_v2t_l[0], debug_length)
            #self.debug(tester, self.ports.debug, debug_length)

            tester.delay(period*0.1)

            # feed through to first output, leave the rest off
            # TODO should maybe leave half open, affects charge sharing?
            #tester.poke(clk.spice_pin, 1)
            #if hasattr(self.ports.clk, '__getitem__'):
            #    for p in self.ports.clk[1:]:
            #        tester.poke(p, 0)
            output = self.signals.out[0]

            limits = self.signals.in_.value
            limit_range = abs(limits[1]-limits[0])
            max_ramp_time = period / 4 # TODO maybe over num_samplers? look as tclk assertion
            if limit_range / abs(s) < max_ramp_time:
                # use the full input range
                start = (limits[0] if s > 0 else limits[1])
                end = (limits[1] if s > 0 else limits[0])

                # time it takes the ramp to get to v
                t_clk = abs((v - start) / s) # TODO abs unnecessary?


                settle_time = self.template.schedule_clk(tester, output, 1, 0.5, values)

                tester.poke(self.ports.in_, start)
                assert period / 2 > t_clk
                tester.delay(period/2 - t_clk) # because of finite rise time of prev line
                tester.poke(self.ports.in_, 0, delay={
                    'type': 'ramp',
                    'start': start,
                    'stop': end,
                    'rate': s,
                    'etol': limit_range/20
                })

                tester.delay(t_clk)
                # sampling is happening right now

                # The read_value call actually does delay long enough for the thing to settle
                if hasattr(self.ports.out, '__getitem__'):
                    p = self.ports.out[0]
                else:
                    p = self.ports.out

                tester.delay(settle_time)
                read = self.template.read_value(tester, p, 0)

                #tester.delay(abs(limit_range/s) - t_clk)
                if settle_time < period/2:
                    tester.delay(period/2 - settle_time)
                print('Using full range for slope', s, ', time', abs(limit_range/s))

            else:
                # use the full ramp time
                # TODO might be outside range
                def clamp(x):
                    if x < min(limits):
                        return min(limits)
                    elif x > max(limits):
                        return max(limits)
                    else:
                        return x
                ss = clamp(v - s * (max_ramp_time/2))
                se = clamp(v + s * (max_ramp_time/2))

                ramp_start_to_sample = abs((ss - v) / s)
                ramp_sample_to_end = abs((se - v) / s)
                tester.poke(self.ports.in_, ss)


                settle_time = self.template.schedule_clk(tester, output, 1, 0.5, values)

                assert ramp_start_to_sample < period / 2
                tester.delay(period / 2 - ramp_start_to_sample)

                tester.poke(self.ports.in_, 0, delay={
                    'type': 'ramp',
                    'start': ss,
                    'stop': se,
                    'rate': s,
                    'etol': abs(se-ss)/20
                })


                tester.delay(ramp_start_to_sample)
                # this is when the clk edge happens, but it's handled
                # by schedule_clk
                #tester.poke(clk, 0)

                tester.delay(settle_time)

                if hasattr(self.ports.out, '__getitem__'):
                    p = self.ports.out[0]
                else:
                    p = self.ports.out
                read = self.template.read_value(tester, p, 0)

                if settle_time < period/2:
                    tester.delay(period/2 - settle_time)
                print('Using full time for slope', s, ', range', ss, se)

            scale = 10**(int(math.log10(1/period)))
            slope_over_scale = s / scale
            print('USING SCALE OF', scale)
            return read, slope_over_scale

        def analysis(self, results):
            read, slope_over_scale = results
            v = self.template.interpret_value(read)
            v_mapped = self.template.temp_inv(v)
            return {'sample_out': v_mapped, 'slope_over_scale': slope_over_scale}


        def post_process(self, results):
            # comment out next line for debug plots
            return results




            vs, ss, jitter, samples, ss_scaled = list(results.values())[:5]
            ss_vpns = [s/1e9 for s in ss]

            #should_be_1 = 0.996
            #alpha = -1.46e-12 * 1e9
            #beta = 8.88e-25 * 1e18
            #gamma = -2.37e-13 * 1e9

            def f(v, s, j=0):
                #should_be_1 = 1.002 + 0.001257 * j
                #alpha = -0.005951 + 0.0002632 * j
                #beta = 0.00009227 + 0.000004501 * j
                #gamma = 0.003088 + (-0.0001502) * j

                should_be_1 = 0.966 + 0.001234 * j
                alpha = -0.03462 + 0.00216 * j
                beta = -0.000004797 + 0.000007284 * j
                gamma = 0.02213 + -0.001437 * j

                return sum([
                    should_be_1 * v,
                    alpha * v*s,
                    beta * s**2,
                    gamma * s
                ])

            import numpy as np
            X = np.arange(min(vs), max(vs), 0.01)
            #Y = np.arange(min(ss_vpns), max(ss_vpns), (max(ss_vpns) - min(ss_vpns))/100)
            J = np.arange(min(jitter), max(jitter), (max(jitter) - min(jitter)) / 100)
            X, J = np.meshgrid(X, J)
            #Z = [[f(x, y) for x,y in zip(xx, yy)] for xx, yy in zip(X, Y)]
            Z = f(X, 0, J)

            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(vs, jitter, samples)
            ax.plot_surface(X, J, Z, alpha=0.5)

            ax.set_xlabel('Input Value (V)')
            #ax.set_ylabel('Input Slope (V / ns)')
            ax.set_ylabel('clk_v2t_l jitter (units)')
            ax.set_zlabel('Measured (V)')
            plt.show()

            return results


        def post_regression(self, regression_results, regression_dataframe):
            return {}
            value = regression_dataframe['value']
            slope = regression_dataframe['slope']
            out = regression_dataframe['sample_out']

            period = float(self.extras['cycle_time'])
            scale = 10 ** (int(math.log10(1 / period)))

            # here we explicity ignore dependence on jitter because it's nominally 0
            results = {k: v[Regression.one_literal] for k, v in regression_results.items()}
            should_be_1 = results['should_be_1']
            alpha = results['alpha_times_scale'] / scale
            beta = results['beta_times_scale2'] / (scale**2)
            gamma = results['gamma_times_scale'] / scale

            pred = (should_be_1 * value
                    + alpha * value * slope
                    + beta * slope ** 2
                    + gamma * slope)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, slope, out, 'g')
            ax.scatter(value, slope, pred, 'b')
            #ax.plot_surface(X, J, Z, alpha=0.5)

            ax.set_title('Green is measured, blue is predicted')
            ax.set_xlabel('Input Value (V)')
            ax.set_ylabel('Input Slope (V / ns)')
            #ax.set_ylabel('clk_v2t_l jitter (units)')
            ax.set_zlabel('Measured (V)')
            plt.show()
            print('done')
            return {}

    #@template_creation_utils.debug
    class SineTest(TemplateMaster.Test):
        num_samples = 1
        parameter_algebra = {}

        def input_domain(self):
            assert False, 'must be converted to new schedule_clk style'
            return []

        def testbench(self, tester, values):
            settle = 0.5 * float(self.extras['approx_settling_time'])
            clk = self.ports.clk[0] if hasattr(self.ports.clk, '__getitem__') else self.ports.clk
            tester.poke(self.ports.in_, 0, delay={
                'type': 'sin',
                'freq': 3e6
            })
            tester.poke(clk, 0, delay={
                'type': 'clock',
                'freq': 10e6
            })
            debug_length = 1
            self.debug(tester, clk, debug_length)
            self.debug(tester, self.ports.out[0], debug_length)
            self.debug(tester, self.ports.in_, debug_length)
            tester.delay(20*settle)
            return []

        def analysis(self, reads):
            return reads


    class ChannelTest(TemplateMaster.Test):
        # sample_out = should_be_1*value + slope * effective_delay
        # effective_delay = alpha*value + beta*slope + gamma*1

        # "sample_adj" is the amount we should add to an ideal sample to model
        # the sampler actually sampling after the clock falls
        # We expect it to be positive when the slope is positive
        # sample_adj = slope * effective_delay
        # effective_delay = alpha*value + beta*slope + gamma*1
        # sample_adj = alpha*value*slope + beta*slope**2 + gamma*slope

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.poked_waveform = False

        #parameter_algebra = {
        #    'sample_out': {'should_be_1': 'value',
        #                   'alpha': 'value*slope',
        #                   'beta': 'slope**2',
        #                   'gamma': 'slope'}
        #}
        parameter_algebra = {
            'sample_adj': {'alpha': 'value*slope',
                           'beta': 'slope**2',
                           'gamma': 'slope',
                           'delta': 'value',
                           'constant': '1'},


            'sample_adj_sp': {
                'alpha_sp': 'value*slope',
                'beta_sp': 'slope**2',
                'gamma_sp': 'slope'
            },
            'sample_adj_sn': {
                'alpha_sn': 'value*slope',
                'beta_sn': 'slope**2',
                'gamma_sn': 'slope'
            },
            'sample_adj_vp': {
                'alpha_vp': 'value*slope',
                'beta_vp': 'slope**2',
                'gamma_vp': 'slope'
            },
            'sample_adj_vn': {
                'alpha_vn': 'value*slope',
                'beta_vn': 'slope**2',
                'gamma_vn': 'slope'
            }
        }
        num_samples = 400

        def input_domain(self):
            jitter_domain = self.template.get_clock_offset_domain()

            return jitter_domain

        def testbench(self, tester, values):
            settle = float(self.extras['approx_settling_time'])
            period = float(self.extras['cycle_time'])
            clk = self.signals.clk[0] if hasattr(self.signals.clk,
                                                 '__getitem__') else self.signals.clk

            if not self.poked_waveform:
                import numpy as np
                data = np.genfromtxt('fixture/test_channel.csv', delimiter=',')
                #ts = [t*settle*0.05 for t in range(200)]
                #vs = [math.sin(t/settle)*0.25+0.95 for t in ts]
                ts, vs = list(data[:, 0]), list(data[:, 1])
                vs_unscaled = data[:, 1]
                limits = self.signals.in_.value
                vs = vs_unscaled*(limits[1] - limits[0]) + limits[0]
                vs = list(vs)
                dt = (data[0][-1] - data[0][0]) / (len(data[0])-1)
                #ts = [dt]*len(vs)

                tester.poke(self.ports.in_, 0, delay={
                    'type': 'future',
                    'waits': [ts[1]]*len(vs),
                    'values': vs
                })
                self.poked_waveform = True

            if hasattr(self.ports.clk, '__getitem__'):
                for p in self.ports.clk[1:]:
                    tester.poke(p, 0)

            debug_length = 1 #wait*10
            self.debug(tester, clk.spice_pin, debug_length)
            self.debug(tester, self.ports.out[0], debug_length)
            self.debug(tester, self.ports.in_, debug_length)

            tester.delay(settle * 0.01)
            print('scheduling clock in ', period)
            self.template.schedule_clk(tester, clk, 0, period, values)


            # TODO this is not the place to declare this
            self.slope_dt = 10e-9#settle * 1e-4
            dt = self.slope_dt
            print('delaying', period-dt)
            tester.delay(period - dt)
            value_early = tester.get_value(self.ports.in_)
            tester.delay(dt)
            value = tester.get_value(self.ports.in_)
            tester.delay(dt)
            value_late = tester.get_value(self.ports.in_)

            if hasattr(self.ports.out, '__getitem__'):
                p = self.ports.out[0]
            else:
                p = self.ports.out
            # TODO is this starting dt too late to see the pulse rising edge?
            print('reading in', settle-dt)
            output = self.template.read_value(tester, p, settle - dt)


            return [value_early, value, value_late], output

        def analysis(self, reads):
            inputs, output = reads
            input_values = [x.value for x in inputs]
            output_value = self.template.interpret_value(output)

            value_early, value, value_late = input_values
            out_mapped = self.template.temp_inv(output_value)
            slope = (value_late - value_early) / (2*self.slope_dt)
            sample_adj = (out_mapped - value)

            v_center = sum(self.signals.in_.value) / 2

            return {'value': value,
                    'slope': slope,
                    'sample_adj': sample_adj,
                    'sample_adj_sp': sample_adj if slope >= 0 else None,
                    'sample_adj_sn': sample_adj if slope < 0 else None,
                    'sample_adj_vp': sample_adj if value >= v_center else None,
                    'sample_adj_vn': sample_adj if value < v_center else None}

        def post_regression(self, regression_results, data):
            return {}
            alpha = PlotHelper.eval_parameter(data, regression_results, 'alpha')
            beta = PlotHelper.eval_parameter(data, regression_results, 'beta')
            gamma = PlotHelper.eval_parameter(data, regression_results, 'gamma')
            delta = PlotHelper.eval_parameter(data, regression_results, 'delta')
            constant = PlotHelper.eval_parameter(data, regression_results, 'constant')
            slope = data['slope']
            value = data['value']
            effective_delay = alpha*value + beta*slope + gamma
            out_adj = effective_delay * slope + delta * value + constant


            sp_mask = ~ data['sample_adj_sp'].isnull()
            alpha_sp = PlotHelper.eval_parameter(data, regression_results, 'alpha_sp')[sp_mask]
            beta_sp = PlotHelper.eval_parameter(data, regression_results, 'beta_sp')[sp_mask]
            gamma_sp = PlotHelper.eval_parameter(data, regression_results, 'gamma_sp')[sp_mask]
            value_sp = data['value'][sp_mask]
            slope_sp = data['slope'][sp_mask]
            effective_delay_sp = alpha_sp*value_sp + beta_sp*slope_sp + gamma_sp
            out_adj_sp = effective_delay_sp * slope_sp

            sn_mask = ~ data['sample_adj_sn'].isnull()
            alpha_sn = PlotHelper.eval_parameter(data, regression_results, 'alpha_sn')[sn_mask]
            beta_sn = PlotHelper.eval_parameter(data, regression_results, 'beta_sn')[sn_mask]
            gamma_sn = PlotHelper.eval_parameter(data, regression_results, 'gamma_sn')[sn_mask]
            value_sn = data['value'][sn_mask]
            slope_sn = data['slope'][sn_mask]
            effective_delay_sn = alpha_sn*value_sn + beta_sn*slope_sn + gamma_sn
            out_adj_sn = effective_delay_sn * slope_sn

            vp_mask = ~ data['sample_adj_vp'].isnull()
            alpha_vp = PlotHelper.eval_parameter(data, regression_results, 'alpha_vp')[vp_mask]
            beta_vp = PlotHelper.eval_parameter(data, regression_results, 'beta_vp')[vp_mask]
            gamma_vp = PlotHelper.eval_parameter(data, regression_results, 'gamma_vp')[vp_mask]
            value_vp = data['value'][vp_mask]
            slope_vp = data['slope'][vp_mask]
            effective_delay_vp = alpha_vp*value_vp + beta_vp*slope_vp + gamma_vp
            out_adj_vp = effective_delay_vp * slope_vp

            vn_mask = ~ data['sample_adj_vn'].isnull()
            alpha_vn = PlotHelper.eval_parameter(data, regression_results, 'alpha_vn')[vn_mask]
            beta_vn = PlotHelper.eval_parameter(data, regression_results, 'beta_vn')[vn_mask]
            gamma_vn = PlotHelper.eval_parameter(data, regression_results, 'gamma_vn')[vn_mask]
            value_vn = data['value'][vn_mask]
            slope_vn = data['slope'][vn_mask]
            effective_delay_vn = alpha_vn*value_vn + beta_vn*slope_vn + gamma_vn
            out_adj_vn = effective_delay_vn * slope_vn

            #plt.plot(value, effective_delay, '*')
            #plt.grid()
            #plt.show()

            adj_measured = data['sample_adj']


            fig = plt.figure()
            plt.plot(slope, effective_delay, 'o')
            plt.grid()
            plt.xlabel('slope')
            plt.ylabel('modeled effective delay')


            fig = plt.figure()
            plt.plot(value, out_adj - adj_measured, '*')
            plt.grid()
            plt.xlabel('input voltage')
            plt.ylabel('predicted_output - measured_output')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, slope, effective_delay)
            ax.set_xlabel('value')
            ax.set_ylabel('slope')
            ax.set_zlabel('effective delay')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, slope, adj_measured)
            ax.scatter(value, slope, out_adj)
            ax.set_xlabel('value')
            ax.set_ylabel('slope')
            ax.set_zlabel('output adjustment')


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, slope, out_adj - adj_measured)
            ax.set_xlabel('value')
            ax.set_ylabel('slope')
            ax.set_zlabel('predicted_output - measured_output')

            pred_high_map = out_adj - adj_measured > 0
            pred_low_map = out_adj - adj_measured < 0


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value[pred_low_map], slope[pred_low_map], adj_measured[pred_low_map])
            ax.scatter(value[pred_high_map], slope[pred_high_map], adj_measured[pred_high_map])
            ax.set_xlabel('value')
            ax.set_ylabel('slope')
            ax.set_zlabel('measured_output')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, slope, out_adj - adj_measured)
            ax.set_xlabel('value')
            ax.set_ylabel('slope')
            ax.set_zlabel('predicted_output - measured_output')


            # not split
            fig = plt.figure()
            plt.plot(slope, adj_measured, 'o')
            plt.plot(slope, out_adj, 'x')
            plt.grid()
            plt.xlabel('slope')
            plt.ylabel('output adjustment')
            plt.legend(['measured', 'predicted'])

            fig = plt.figure()
            plt.plot(slope, out_adj - adj_measured, '*')
            plt.grid()
            plt.xlabel('slope')
            plt.ylabel('error (pred - meas)')

            fig = plt.figure()
            plt.plot(value, adj_measured, 'o')
            plt.plot(value, out_adj, 'x')
            plt.grid()
            plt.xlabel('voltage')
            plt.ylabel('output adjustment')
            plt.legend(['measured', 'predicted'])

            fig = plt.figure()
            plt.plot(value, out_adj - adj_measured, '*')
            plt.grid()
            plt.xlabel('voltage')
            plt.ylabel('error (pred - meas)')

            # split by voltage
            plt.figure()
            plt.plot(slope, adj_measured, 'o')
            plt.plot(slope_vp, out_adj_vp, 'x')
            plt.plot(slope_vn, out_adj_vn, 'x')
            plt.xlabel('slope')
            plt.ylabel('output adjustment')
            plt.legend(['measured', 'predicted (pos voltage)', 'predicted (neg voltage)'])
            plt.grid()

            plt.figure()
            plt.plot(slope_vp, out_adj_vp - (adj_measured[vp_mask]), '*')
            plt.plot(slope_vn, out_adj_vn - (adj_measured[vn_mask]), '*')
            plt.xlabel('slope')
            plt.ylabel('error (pred - meas)')
            plt.grid()

            plt.figure()
            plt.plot(value, adj_measured, 'o')
            plt.plot(value_vp, out_adj_vp, 'x')
            plt.plot(value_vn, out_adj_vn, 'x')
            plt.xlabel('voltage')
            plt.ylabel('output adjustment')
            plt.legend(['measured', 'predicted (pos voltage)', 'predicted (neg voltage)'])
            plt.grid()

            plt.figure()
            plt.plot(value_vp, out_adj_vp - (adj_measured[vp_mask]), '*')
            plt.plot(value_vn, out_adj_vn - (adj_measured[vn_mask]), '*')
            plt.xlabel('voltage')
            plt.ylabel('error (pred - meas)')
            plt.grid()


            # split by slope
            plt.figure()
            plt.plot(slope, adj_measured, 'o')
            plt.plot(slope_sp, out_adj_sp, 'x')
            plt.plot(slope_sn, out_adj_sn, 'x')
            plt.xlabel('slope')
            plt.ylabel('output adjustment')
            plt.legend(['measured', 'predicted (pos slope)', 'predicted (neg slope)'])
            plt.grid()

            plt.figure()
            plt.plot(slope_sp, out_adj_sp - (adj_measured[sp_mask]), '*')
            plt.plot(slope_sn, out_adj_sn - (adj_measured[sn_mask]), '*')
            plt.xlabel('slope')
            plt.ylabel('error (pred - meas)')
            plt.show()

            plt.figure()
            plt.plot(value, adj_measured, 'o')
            plt.plot(value_sp, out_adj_sp, 'x')
            plt.plot(value_sn, out_adj_sn, 'x')
            plt.xlabel('voltage')
            plt.ylabel('output adjustment')
            plt.legend(['measured', 'predicted (pos slope)', 'predicted (neg slope)'])
            plt.grid()

            plt.figure()
            plt.plot(value_sp, out_adj_sp - (adj_measured[sp_mask]), '*')
            plt.plot(value_sn, out_adj_sn - (adj_measured[sn_mask]), '*')
            plt.xlabel('voltage')
            plt.ylabel('error (pred - meas)')
            plt.show()
            return {}


    class DelayTest(TemplateMaster.Test):
        parameter_algebra = {
            'output_adj': {'alpha': 'value*slope',
                           'beta': 'slope**2',
                           'gamma': 'slope'},

            'delay': {'A': 'value',
                      'B': 'slope',
                      'C': '1'},
            'delay_ps': {'A_ps': 'value',
                         'B_ps': 'slope',
                         'C_ps': '1'},
            'delay_ns': {'A_ns': 'value',
                         'B_ns': 'slope',
                         'C_ns': '1'},
            'delay_pv': {'A_pv': 'value',
                         'B_pv': 'slope',
                         'C_pv': '1'},
            'delay_nv': {'A_nv': 'value',
                         'B_nv': 'slope',
                         'C_nv': '1'},
        }
        num_samples = 50 # was 400

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.poked_waveform = False
            self.blocks = []

            assert 'channel_info' in self.template.extras
            info = self.template.extras['channel_info']
            file_path = info['file_path']
            bit_freq = float(info['bit_freq'])
            voltage_range = ast.literal_eval(info['voltage_range'])

            total_time = self.num_samples * self.template.extras['cycle_time']
            total_time *= 1.5
            time_stretch = float(info.get('fake_channel_time_stretch', 1))



            channel_data = ChannelUtil.get_channel_data(
                file_path,
                bit_freq / time_stretch,
                bit_freq / time_stretch * 100,
                total_time * time_stretch,
                debug=False
            )
            time = channel_data[:, 0]
            data = channel_data[:, 1]
            data = data*(voltage_range[1]-voltage_range[0]) + voltage_range[0]
            time /= time_stretch
            self.channel_data = (time, data)

        def input_domain(self):
            return []

        def testbench(self, tester, values):
            #settle = float(self.extras['approx_settling_time'])
            period = float(self.extras['cycle_time'])
            #clk = self.signals.clk[0] if hasattr(self.signals.clk,
            #                                     '__getitem__') else self.signals.clk

            if not self.poked_waveform:
                #import numpy as np
                #data = np.genfromtxt('fixture/test_channel.csv', delimiter=',')
                ##ts = [t*settle*0.05 for t in range(200)]
                ##vs = [math.sin(t/settle)*0.25+0.95 for t in ts]
                #ts, vs = list(data[:, 0]), list(data[:, 1])
                #vs_unscaled = data[:, 1]
                #limits = self.signals.in_.value
                #vs = vs_unscaled*(limits[1] - limits[0]) + limits[0]
                #vs = list(vs)
                #dt = (data[0][-1] - data[0][0]) / (len(data[0])-1)
                ##ts = [dt]*len(vs)
                ts, vs = self.channel_data

                tester.poke(self.ports.in_, 0, delay={
                    'type': 'future',
                    'waits': [ts[1]]*len(vs),
                    'values': vs
                })
                self.poked_waveform = True

            #if hasattr(self.ports.clk, '__getitem__'):
            #    for p in self.ports.clk[1:]:
            #        tester.poke(p, 0)

            debug_length = 1 #wait*10
            self.debug(tester, self.ports.clk[0], debug_length)
            #self.debug(tester, self.ports.clk[1], debug_length)
            if hasattr(self.ports, 'debug'):
                self.debug(tester, self.ports.debug, debug_length)

            # TODO re-enable this debug
            #self.debug(tester, self.ports.out[0], debug_length)
            self.debug(tester, self.ports.in_, debug_length)

            # TODO why was that delay there? simulator issues?
            # Now we need it so we aren't locked to channel freq
            tester.delay(period * 0.01)

            time_sample_to_read = self.template.schedule_clk(tester, self.signals.out[0], 1, 0.5, values)

            block = tester.get_value(self.ports.in_,
                             params={'style': 'block', 'duration': 1*period})



            # TODO this breaks if settle > period
            self.slope_dt = period / 100 # 10e-9
            tester.delay(period/2 - self.slope_dt)
            v_early = tester.get_value(self.ports.in_)
            tester.delay(self.slope_dt)
            v_exact = tester.get_value(self.ports.in_)
            tester.delay(self.slope_dt)
            v_late = tester.get_value(self.ports.in_)
            assert time_sample_to_read > self.slope_dt
            tester.delay(time_sample_to_read - self.slope_dt)

            if hasattr(self.ports.out, '__getitem__'):
                p = self.ports.out[0]
            else:
                p = self.ports.out
            #output = tester.get_value(p)
            output = self.template.read_value(tester, p, 0)
            if period / 2 - time_sample_to_read > 0:
                tester.delay(period/2 - time_sample_to_read)

            # delay about 10% a period so we are not locked to the sampling
            # frequency, and use an irrational number so not locked at any ratio
            unlock = (5**.5-1)/12
            tester.delay(period*unlock)

            return block, v_early, v_exact, v_late, output

        def analysis(self, reads):
            block, v_early, v_exact, v_late = [x.value for x in reads[:-1]]
            v_early, v_exact, v_late = [float(x) for x in [v_early, v_exact, v_late]]
            self.blocks.append(scipy.interpolate.interp1d(*block))
            output = self.template.interpret_value(reads[-1])
            # TODO I don't understand why that cast to float is necessary
            out_mapped = float(self.template.temp_inv(output))

            slope = (v_late - v_early) / (2 * self.slope_dt)

            # block read is 1 period, clk falls right in the middle
            period = float(self.extras['cycle_time'])
            def search(forward, rising):
                try:
                    es = domain_read.find_edge_spice(block[0], block[1],
                            period / 2, out_mapped, forward=forward, rising=rising)
                    return es[0]
                except domain_read.EdgeNotFoundError:
                    return None

            edges = [search(False, False),
                     search(False, True),
                     search(True, False),
                     search(True, True)]




            def closer(a, b):
                if a is None:
                    return b
                elif b is None:
                    return a
                return a if abs(a) < abs(b) else b
            closest = functools.reduce(closer, edges)


            #if closest is not None and (not (0.7e-8 < closest < 1.0e-8)):
            #    closest = None
            #if closest is not None and (not (0 < closest < 1e-7)):
            #    closest = None


            #if closest == None:
            #    plt.plot(block[0] - period/2, block[1], '-+')
            #    plt.plot([-period, period], [out_mapped, out_mapped])
            #    for e in edges:
            #        if e != None:
            #            plt.plot([e, e], [0.6, 0.7])
            #    plt.show()


            #ps_mask = slope > 0
            #ns_mask = slope < 0
            #closest_ps = [c if x else None for c, x in zip(closest, ps_mask)]
            #closest_ns = [c if x else None for c, x in zip(closest, ns_mask)]
            #closest_ps, closest_ns = np.array(closest_ps), np.array(closest_ns)
            closest_ps = closest if slope >= 0 else None
            closest_ns = closest if slope <  0 else None
            v_split = sum(self.signals.in_.value) / 2
            closest_pv = closest if v_exact >= v_split else None
            closest_nv = closest if v_exact <  v_split else None


            return {'value': v_exact, 'slope': slope, 'output': out_mapped,
                    'output_adj': out_mapped- v_exact,
                    'delay': closest,
                    'delay_ps': closest_ps,
                    'delay_ns': closest_ns,
                    'delay_pv': closest_pv,
                    'delay_nv': closest_nv,
                    'output_raw_debug': output
                    }

        def post_regression(self, regression_models, data):
            return {}
            value = data['value']
            slope = data['slope']
            #output = data['output']
            output_adj = data['output_adj']
            delay = data['delay']
            period = float(self.extras['cycle_time'])

            alpha = PlotHelper.eval_parameter(data, regression_models, 'alpha')
            beta = PlotHelper.eval_parameter(data, regression_models, 'beta')
            gamma = PlotHelper.eval_parameter(data, regression_models, 'gamma')
            A = PlotHelper.eval_parameter(data, regression_models, 'A')
            B = PlotHelper.eval_parameter(data, regression_models, 'B')
            C = PlotHelper.eval_parameter(data, regression_models, 'C')
            delay_v1 = alpha * value + beta * slope + gamma
            delay_v2 = A * value + B * slope + C
            projected_adj_v1 = delay_v1 * slope
            projected_adj_v2 = delay_v2 * slope

            # ps ns
            A_ps = PlotHelper.eval_parameter(data, regression_models, 'A_ps')
            B_ps = PlotHelper.eval_parameter(data, regression_models, 'B_ps')
            C_ps = PlotHelper.eval_parameter(data, regression_models, 'C_ps')
            delay_v2_ps = A_ps * value + B_ps * slope + C_ps
            projected_adj_v2_ps = delay_v2_ps * slope
            A_ns = PlotHelper.eval_parameter(data, regression_models, 'A_ns')
            B_ns = PlotHelper.eval_parameter(data, regression_models, 'B_ns')
            C_ns = PlotHelper.eval_parameter(data, regression_models, 'C_ns')
            delay_v2_ns = A_ns * value + B_ns * slope + C_ns
            projected_adj_v2_ns = delay_v2_ns * slope
            # pv nv
            A_pv = PlotHelper.eval_parameter(data, regression_models, 'A_pv')
            B_pv = PlotHelper.eval_parameter(data, regression_models, 'B_pv')
            C_pv = PlotHelper.eval_parameter(data, regression_models, 'C_pv')
            delay_v2_pv = A_pv * value + B_pv * slope + C_pv
            projected_adj_v2_pv = delay_v2_pv * slope
            A_nv = PlotHelper.eval_parameter(data, regression_models, 'A_nv')
            B_nv = PlotHelper.eval_parameter(data, regression_models, 'B_nv')
            C_nv = PlotHelper.eval_parameter(data, regression_models, 'C_nv')
            delay_v2_nv = A_nv * value + B_nv * slope + C_nv
            projected_adj_v2_nv = delay_v2_nv * slope

            def resample(delays):
                def resample_one(i, dt):
                    block = self.blocks[i]
                    try:
                        return block(period/2 + dt)
                    except ValueError:
                        return None
                xs = [resample_one(i, dt) for i, dt in enumerate(delays)]
                return np.array(xs)
            resampled_adj_v1 = resample(delay_v1) - value
            resampled_adj_v2 = resample(delay_v2) - value



            # delay plots
            #plt.figure()
            plt.plot(slope, delay, '*')
            plt.plot(slope, delay_v2, 'x')
            plt.legend(['Measured ideal delay', 'Modeled delay'])
            plt.title('Version 2: effective delay based on incoming waveform')
            plt.xlabel('slope')
            plt.ylabel('delay')
            plt.grid()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, slope, delay)
            ax.scatter(value, slope, delay_v2)
            ax.set_xlabel('value')
            ax.set_ylabel('slope')
            ax.set_zlabel('delay')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, slope, delay)
            ax.scatter(value, slope, delay_v2_ps)
            ax.scatter(value, slope, delay_v2_ns)
            ax.set_xlabel('value')
            ax.set_ylabel('slope')
            ax.set_zlabel('delay (split by slope)')


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, slope, delay)
            ax.scatter(value, slope, delay_v2_pv)
            ax.scatter(value, slope, delay_v2_nv)
            ax.set_xlabel('value')
            ax.set_ylabel('slope')
            ax.set_zlabel('delay (split by value)')


            # split on slope stuff

            plt.figure()
            #plt.clf()
            plt.plot(slope, delay, '*')
            plt.plot(slope, delay_v2_ps, 'x')
            plt.plot(slope, delay_v2_ns, '+')
            plt.legend(['Measured delay', 'ps modeled delay', 'ns modeled delay'])
            plt.title('Version 2: effective delay based on resampling')
            plt.xlabel('slope')
            plt.ylabel('delay')
            plt.grid()


            # final 6 plots below
            plt.figure()
            plt.plot(slope, output_adj, '*')
            plt.plot(slope, projected_adj_v1, 'x')
            plt.plot(slope, resampled_adj_v1, '+')
            plt.legend(['Measured adjustment', 'Modeled projected adjustment', 'Modeled resampled adjustment'])
            plt.title('Version 1: effective delay based on projecting slope')
            plt.xlabel('slope')
            plt.ylabel('sample adjustment')
            plt.grid()

            plt.figure()
            plt.plot(value, output_adj, '*')
            plt.plot(value, projected_adj_v1, 'x')
            plt.plot(value, resampled_adj_v1, '+')
            plt.legend(['Measured adjustment', 'Modeled projected adjustment', 'Modeled resampled adjustment'])
            plt.title('Version 1: effective delay based on projecting slope')
            plt.xlabel('value')
            plt.ylabel('sample adjustment')
            plt.grid()

            plt.figure()
            # empty to take up a color?
            plt.plot([], [], '*')
            plt.plot(slope, projected_adj_v1 - output_adj, 'x')
            plt.plot(slope, resampled_adj_v1 - output_adj, '+')
            #plt.legend(['Measured adjustment', 'Modeled projected adjustment', 'Modeled resampled adjustment'])
            plt.title('Version 1: effective delay based on projecting slope')
            plt.xlabel('slope')
            plt.ylabel('sample model error')
            plt.grid()

            plt.figure()
            plt.plot(slope, output_adj, '*')
            plt.plot(slope, projected_adj_v2, 'x')
            plt.plot(slope, resampled_adj_v2, '+')
            plt.legend(['Measured adjustment', 'Modeled projected adjustment', 'Modeled resampled adjustment'])
            plt.title('Version 2: effective delay based on incoming waveform')
            plt.xlabel('slope')
            plt.ylabel('sample adjustment')
            plt.grid()

            plt.figure()
            plt.plot(value, output_adj, '*')
            plt.plot(value, projected_adj_v2, 'x')
            plt.plot(value, resampled_adj_v2, '+')
            plt.legend(['Measured adjustment', 'Modeled projected adjustment', 'Modeled resampled adjustment'])
            plt.title('Version 2: effective delay based on incoming waveform')
            plt.xlabel('value')
            plt.ylabel('sample adjustment')
            plt.grid()

            plt.figure()
            # empty to take up a color?
            plt.plot([], [], '*')
            plt.plot(slope, projected_adj_v2 - output_adj, 'x')
            plt.plot(slope, resampled_adj_v2 - output_adj, '+')
            #plt.legend(['Measured adjustment', 'Modeled projected adjustment', 'Modeled resampled adjustment'])
            plt.title('Version 2: effective delay based on incoming waveform')
            plt.xlabel('slope')
            plt.ylabel('sample model error')
            plt.grid()
            plt.show()
            return {}


    class KickbackTest(TemplateMaster.Test):
        parameter_algebra = {
            'kickback': {'alpha': 'value',
                         'beta': 'old_value',
                         'gamma': '1'},
        }
        num_samples = 100

        #def __init__(self, *args, **kwargs):
        #    super().__init__(*args, **kwargs)
        #    self.poked_waveform = False
        #    self.blocks = []

        def input_domain(self):
            limits = self.signals.in_.value
            current = signals.create_input_domain_signal('current', limits)
            two_ago = signals.create_input_domain_signal('two_ago', limits)
            return [current, two_ago]

        def testbench(self, tester, values):
            period = float(self.extras['cycle_time'])
            two_ago = values['two_ago']
            current = values['current']

            debug_length = 1 #wait*10
            self.debug(tester, self.ports.clk[0], debug_length)
            #self.debug(tester, self.ports.clk[1], debug_length)
            try:
                self.debug(tester, self.signals.from_circuit_name('debug').spice_pin, debug_length)
            except KeyError:
                pass
            # TODO reenable this debug
            #self.debug(tester, self.ports.out[0], debug_length)
            self.debug(tester, self.ports.in_, debug_length)


            time_sample_to_read = self.template.schedule_clk(tester, self.signals.out[0], 2, 0.5, values)


            #tester.delay(period*1.5)
            #tester.poke(self.ports.in_, 0)
            #tester.delay(period*0.005)
            #tester.poke(self.ports.in_, 1)
            #tester.delay(period*0.495)
            #tester.poke(self.ports.in_, 1)
            #tester.delay(period*0.005)
            #tester.poke(self.ports.in_, 0)
            #tester.delay(period*0.995)


            # 2 periods are about to play
            # our main clock edge is falling in period*1.5
            # we want to feed the other value into the edge at period*0.75
            tester.poke(self.ports.in_, 0)
            tester.delay(0.875*period)
            #tester.delay(0.5*period) # debug
            tester.poke(self.ports.in_, 0)
            tester.delay(0.005*period)
            tester.poke(self.ports.in_, two_ago)
            tester.delay(0.120*period)

            # end of 1st period

            tester.delay(0.125*period)
            tester.poke(self.ports.in_, two_ago)
            tester.delay(0.005*period)
            tester.poke(self.ports.in_, 0)
            tester.delay(0.120*period)

            # 1.25 periods in

            tester.delay(0.125*period)
            tester.poke(self.ports.in_, 0)
            tester.delay(0.005*period)
            tester.poke(self.ports.in_, current)
            tester.delay(0.120*period)

            # sampling edge falls right here, 1.5 periods in
            tester.delay(time_sample_to_read)
            #meas = tester.get_value(self.ports.out[0])
            meas = self.template.read_value(tester, self.ports.out[0], 0)
            tester.delay(0.500*period - time_sample_to_read)

            # end of 2 periods

            tester.poke(self.ports.in_, current)
            tester.delay(0.005*period)
            tester.poke(self.ports.in_, 0)
            tester.delay(0.245*period)


            return [current, two_ago, meas]

        def analysis(self, reads):
            current, two_ago, meas_gv = reads
            meas = meas_gv.value
            meas_mapped = float(self.template.temp_inv(meas))

            return {'value': current,
                    'old_value': two_ago,
                    'kickback': meas_mapped - current,
                    'meas': meas_mapped}

        def post_regression(self, models, data):
            return {}
            value = PlotHelper.eval_factor(data, 'current')
            old_value = PlotHelper.eval_factor(data, 'two_ago')
            kickback = PlotHelper.eval_factor(data, 'kickback')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
            ax.scatter(value, old_value, kickback)
            ax.set_xlabel('value')
            ax.set_ylabel('old_value')
            ax.set_zlabel('kickback')
            plt.show()
            return {}


    tests = [
             StaticNonlinearityTest,
             #ApertureTest,
             #ChannelTest,
             #SineTest,
             DelayTest,
             KickbackTest
            ]

