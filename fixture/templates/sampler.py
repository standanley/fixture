from fixture import TemplateMaster
from fixture import template_creation_utils
from fixture import signals
import math


class SamplerTemplate(TemplateMaster):
    required_ports = ['in_', 'clk', 'out']
    required_info = {
        'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
    }

    def __init__(self, *args, **kwargs):
        # Some magic constants, maybe pull these from config?
        # NOTE this is before super() because it is used for Test instantiation
        self.nonlinearity_points = 3 # 31

        # we have to do this before getting input domains, which happens
        # in the call to super
        extras = args[3]
        if 'clks' in extras:
            settle = float(extras['clks']['unit']) * float(extras['clks']['period'])
            extras['approx_settling_time'] = settle

        super().__init__(*args, **kwargs)

        if 'clks' in self.extras:
            settle = float(self.extras['clks']['unit']) * float(self.extras['clks']['period'])
            self.extras['approx_settling_time'] = settle

            clks = self.extras['clks']
            clks = {k: v for k, v in clks.items() if (k != 'unit' and k != 'period')}
            domain = []
            for clk, v in clks.items():
                if 'max_jitter' in v:
                    x = v['max_jitter']
                    self.signals.add_signal(signals.SignalIn(
                        (-x, x),
                        'bit',
                        True,
                        False,
                        None,
                        None,
                        clk+'_jitter'
                    ))

        # NOTE this must be after super() because it needs ports to be defined

        if not hasattr(self.ports.clk, '__getitem__'):
            #self.ports.mapping['clk'] = [self.ports.mapping['clk']]
            #self.ports.mapping['out'] = [self.ports.mapping['out']]
            pass
        else:
            assert len(self.ports.out) == len(self.ports.clk), \
                'Must have one clock for each output!'


    def read_value(self, tester, port, wait):
        tester.delay(wait)
        return tester.get_value(port)

    def interpret_value(self, read):
        return read.value

    #def schedule_clk(self, tester, port, value, wait):
    #    tester.poke(port, value, delay={'type': 'future', 'wait': wait})

    def get_clock_offset_domain(self):
        return []

    def schedule_clk(self, tester, main, value, wait, jitters={}):
        if 'clks' not in self.extras:
            # TODO make this better please
            tester.poke(main.spice_pin, 0, delay={
                'type': 'future',
                'waits': [wait],
                'values': [value]
            })
            return

        clks = self.extras['clks']

        for s_ignore in self.signals.ignore:
            if s_ignore is None:
                continue
            if s_ignore.spice_name not in clks:
                tester.poke(s_ignore.spice_pin, 0)

        if isinstance(self.signals.clk, list):
            clk_list = self.signals.clk
            num_samplers = len(clk_list)
            desired_sampler = clk_list.index(main)
        else:
            #main = self.get_name_circuit(port)
            num_samplers = 1
            desired_sampler = 0
        unit = float(clks['unit'])
        period = clks['period']
        clks = {self.signals.from_spice_name(k): v for k,v in clks.items()
                if (k!='unit' and k!='period')}


        # To take our measurement we will play through played_periods,
        # and then take a measurement based on the falling edge of the main
        # clock during the measured_period (zero-indexed)
        played_periods = 2
        measured_period = 1

        main_period_start_time = [t for t,v in clks[main].items() if v==1][0]

        # shift the user-given period such that the main clock's
        # rising edge just happened
        # Presumably the sampling edge is now in the middle of the period

        for i in range(num_samplers):
            offset = ((i - desired_sampler + num_samplers) % num_samplers) / num_samplers
            period_start_time = (main_period_start_time + period * offset) % period

            clks_transform = {}

            for clk in clks:
                temp = []
                name = clk[0].spice_name if isinstance(clk, list) else clk.spice_name
                jitter_name = name + '_jitter'
                jitter = jitters.get(jitter_name, 0)
                for p in range(played_periods):
                    for time, val in clks[clk].items():
                        if time == 'max_jitter':
                            continue
                        time_transform = time + (period - period_start_time)
                        if time_transform > period:
                            time_transform -= period
                        #time_transform *= unit
                        time_transform_shift = time_transform + (p - measured_period) * period
                        temp.append((time_transform_shift + jitter, val))
                clks_transform[clk] = sorted(temp)

            # shift that one period s.t. the falling edge of the main clk
            # happens after exactly "wait"
            # simply ignore any edges that would've been in the past
            # shift is the time in seconds from now until the period start
            shift = wait - period_start_time * unit
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
                tester.poke(current_clk.spice_pin, 0, delay={
                    'type': 'future',
                    'waits': waits,
                    'values': values
                })


    #@template_creation_utils.debug
    class StaticNonlinearityTest(TemplateMaster.Test):
        num_samples = 10#3

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
            settle = float(self.extras['approx_settling_time'])
            wait = 2 * settle
            clk = self.signals.clk[0] if hasattr(self.signals.clk, '__getitem__') else self.signals.clk
            assert isinstance(clk, signals.SignalIn)

            # To see debug plots, also uncomment debug decorator for this class
            np = self.template.nonlinearity_points
            debug_time = np * wait * 22
            self.debug(tester, clk.spice_pin, debug_time)
            self.debug(tester, self.ports.out, debug_time)
            self.debug(tester, self.ports.in_, debug_time)

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

            tester.delay(wait*0.2)

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
            for i in range(num):
                dc = limits[0] + i * (limits[1] - limits[0]) / (num-1)
                tester.poke(clk.spice_pin, 1)
                tester.poke(self.ports.in_, dc)

                self.template.schedule_clk(tester, clk, 0, wait, values)
                tester.delay(wait)

                # delays time "wait" for things to settle before reading
                read = self.template.read_value(tester, p, wait)

                # small delay so value is not changed by start of next test
                tester.delay(wait * 0.1)
                results.append((dc, read))

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

    #@template_creation_utils.debug
    class ApertureTest(TemplateMaster.Test):

        # sample_out = should_be_1*value + slope * effective_delay
        # effective_delay = alpha*value + beta*slope + gamma*1

        parameter_algebra = {
            'sample_out': {'should_be_1': 'value',
                           'alpha_times_scale': 'value*slope_over_scale',
                           'beta_times_scale2': 'slope_over_scale**2',
                           'gamma_times_scale': 'slope_over_scale'}
        }
        num_samples = 2

        def input_domain(self):
            limits = self.signals.from_template_name('in_').value
            settle = 0.5 * float(self.extras['approx_settling_time'])
            max_slope = 50 * (limits[1]-limits[0]) / settle

            v = signals.create_input_domain_signal('value', limits)
            s = signals.create_input_domain_signal('slope', (-max_slope, max_slope))
            jitter_domain = self.template.get_clock_offset_domain()

            return [v, s] + jitter_domain

        def testbench(self, tester, values):
            #print('Chose jitter value', values['clk_v2t_l_jitter'], 'in dynamic test')
            settle = float(self.extras['approx_settling_time'])
            wait = 1 * settle
            v = values['value']
            s = values['slope']
            clk = self.signals.clk[0] if hasattr(self.signals.clk, '__getitem__') else self.signals.clk
            assert isinstance(clk, signals.SignalIn)

            # To see debug plots, also uncomment debug decorator for this class
            np = self.template.nonlinearity_points
            debug_length = np * wait * 30
            self.debug(tester, clk.spice_pin, debug_length)
            self.debug(tester, self.ports.out, debug_length)
            self.debug(tester, self.ports.in_, debug_length)
            #self.debug(tester, self.template.dut.clk_v2t_l[0], debug_length)

            tester.delay(wait*0.1)

            # feed through to first output, leave the rest off
            # TODO should maybe leave half open, affects charge sharing?
            tester.poke(clk.spice_pin, 1)
            if hasattr(self.ports.clk, '__getitem__'):
                for p in self.ports.clk[1:]:
                    tester.poke(p, 0)

            limits = self.signals.in_.value
            limit_range = abs(limits[1]-limits[0])
            max_ramp_time = settle/5 # TODO
            if limit_range / abs(s) < max_ramp_time:
                # use the full input range
                start = (limits[0] if s > 0 else limits[1])
                end = (limits[1] if s > 0 else limits[0])

                t_clk = abs((v - start) / s) # TODO abs unnecessary?
                self.template.schedule_clk(tester, clk, 0, t_clk + wait*2, values)

                tester.poke(self.ports.in_, start)
                tester.delay(wait*2) # because of finite rise time of prev line
                tester.poke(self.ports.in_, 0, delay={
                    'type': 'ramp',
                    'start': start,
                    'stop': end,
                    'rate': s,
                    'etol': limit_range/20
                })

                tester.delay(t_clk)

                # The read_value call actually does delay long enough for the thing to settle
                if hasattr(self.ports.out, '__getitem__'):
                    p = self.ports.out[0]
                else:
                    p = self.ports.out
                read = self.template.read_value(tester, p, wait)

                tester.delay(abs(limit_range/s) - t_clk)
                tester.delay(wait * 1)
                print('Using full range for slope', s, ', time', abs(limit_range/s))
                pass

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


                tester.poke(self.ports.in_, ss)
                t_delay = abs((ss-v)/s)
                self.template.schedule_clk(tester, clk, 0, t_delay + wait*2, values)
                tester.delay(wait) # because of finite rise time of prev line
                tester.poke(self.ports.in_, 0, delay={
                    'type': 'ramp',
                    'start': ss,
                    'stop': se,
                    'rate': s,
                    'etol': abs(se-ss)/20
                })


                tester.delay(t_delay)
                # this is when the clk edge happens, but it's handled
                # by schedule_clk
                #tester.poke(clk, 0)
                if hasattr(self.ports.out, '__getitem__'):
                    p = self.ports.out[0]
                else:
                    p = self.ports.out
                read = self.template.read_value(tester, p, wait)

                tester.delay(max_ramp_time/2)
                print('Using full time for slope', s, ', range', ss, se)

            scale = 10**(int(math.log10(1/settle)))
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
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(vs, jitter, samples)
            ax.plot_surface(X, J, Z, alpha=0.5)

            ax.set_xlabel('Input Value (V)')
            #ax.set_ylabel('Input Slope (V / ns)')
            ax.set_ylabel('clk_v2t_l jitter (units)')
            ax.set_zlabel('Measured (V)')
            plt.show()

            return results

    #@template_creation_utils.debug
    class SineTest(TemplateMaster.Test):
        num_samples = 1
        parameter_algebra = {}

        def input_domain(self):
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


    tests = [#SineTest,
             StaticNonlinearityTest,
             ApertureTest]

