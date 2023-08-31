from fixture import TemplateMaster
from fixture.plot_helper import plt, PlotHelper
from fixture.signals import create_input_domain_signal


class PhaseBlenderTemplate_C(TemplateMaster):
    required_ports = ['in_a', 'in_b', 'out']
    required_info = {
        'phase_offset_range': 'phase offset between in_a and in_b, must be between 0 and 1',
        'frequency': 'Input clock frequency (Hz)'
    }

    #@debug
    class Test1(TemplateMaster.Test):
        #parameter_algebra = {
        #    'out_delay': {'gain':'in_phase_delay', 'offset':'1'}
        #    #'out_delay': {'offset': '1'}
        #    #'out_delay': {'offset_a': 'sel_therm', 'offset_b': '1'}
        #}
        analysis_outputs = ['out_delay']
        parameters = ['gain', 'offset']
        parameter_algebra = {
            'out_delay': 'gain * in_phase_delay + offset'
        }
        num_samples = 300


        def input_domain(self):
            ## offset range is in units of "periods", so 0.5 means in_a and in_b are in quadrature
            offset_range = self.extras.get('phase_offset_range', (0, .5))
            freq = float(self.extras['frequency'])
            offset_delay_range = tuple((offset / freq for offset in offset_range))
            '''
            diff = RealIn(offset_delay_range)
            # TODO make this part of the instantiation of RealIn
            diff.name = 'in_phase_delay'
            # could make a new test vector with same params as sel, or just use sel itself
            # new_sel = Array(len(self.sel), BinaryAnalog)
            '''
            diff = create_input_domain_signal('in_phase_delay', offset_delay_range)
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

            # the clock starts just after its falling edge
            tester.poke(self.ports.in_a, 0, delay={
                'freq': freq,
                })
            tester.delay(in_phase_delay)
            tester.poke(self.ports.in_b, 0, delay={
                'freq': freq,
                })
            tester.delay(1/freq - in_phase_delay)

            # 2 run cycles should be enough but... for safety
            run_cycles = 3
            settle_cycles = 3

            # we want to turn off just a touch before the next rising edge
            # to avoid ambiguity if 2 edges fall on top of each other
            turn_off_shift = 0.01
            tester.delay((run_cycles - 1 - turn_off_shift) / freq + in_phase_delay)

            tester.poke(self.ports.in_a, 0)
            tester.poke(self.ports.in_b, 0)


            # now we wait for the output edge to settle
            tester.delay((settle_cycles + turn_off_shift)/freq - in_phase_delay)


            in_b_last = tester.get_value(self.ports.in_b, params={
                'style': 'edge',
                'forward': False,
                'count': 1,
                'rising': True
            })
            out_last = tester.get_value(self.ports.out, params={
                'style': 'edge',
                'forward': False,
                'count': 1,
                'rising': True
            })

            # wait a touch longer so our read doesn't happen on the next one's
            # initial rising edge
            tester.delay(0.1 / freq)
            return [in_b_last, out_last, in_phase_delay]

        def analysis(self, reads):
            freq = float(self.extras['frequency'])

            # delta because it's time from <read request> to <edge> (always negative)
            in_b_delta = reads[0].value[0]
            out_delta = reads[1].value[0]
            in_phase_delay = reads[2]

            ## fix for unintentional wrapping below 0
            #period = 1 / freq
            #if out_delay > 0.9 * period:
            #    out_delay -= period

            delay_seconds = out_delta - in_b_delta + in_phase_delay
            ret = {'out_delay': delay_seconds}
            #print('returning outdelay of', delay_seconds*freq)
            return ret

        def post_process(self, results):
            # We don't need to do any post-process phase unwrapping! yay
            return {}#results

        def post_regression(self, results, data):
            return {}
            # This has not been updated to new post_regression signature
            import numpy as np

            def new_pred_fun(x, dt, abcdef):
                a, b, c, d, e, f = abcdef
                alpha = a * x + b
                beta = c * x + d
                gamma = e * x + f
                if alpha < dt:
                    #print('CASE AAA', alpha, dt + (1 - dt/alpha) * beta)
                    return alpha + gamma
                else:
                    #print('CASE B', alpha, dt + (1 - dt/alpha) * beta)
                    return dt + (1 - dt/alpha) * beta + gamma


            model = list(regression_models.values())[0]
            data = model.model.data
            vectors = data.exog
            measured = data.endog
            predictions = model.predict()
            my_predictions = [sum(vector) for vector in vectors]

            def get_color(vector):
                def temp(x):
                    return .2 + .8*x
                return [temp(x) for x in vector[1:5]]
            c = [get_color(vector) for vector in vectors]

            xs = [sum(v[len(v)//2+1:]) for v in vectors]
            bits = [v[len(v)//2+1:] for v in vectors]
            ys = [v[0] for v in vectors]
            zs = measured
            xs = np.array(xs)
            ys = np.array(ys)
            zs = np.array(zs)

            # linear regression
            # lin_pred = a*(x*y) + b*x + c*y + d
            dat = np.zeros((len(xs), 4))
            dat[:, 0] = (xs * ys)
            dat[:, 1] = (xs)
            dat[:, 2] = (ys)
            dat[:, 3] = np.ones((len(xs)))
            lin = np.linalg.inv(dat.T @ dat) @ dat.T @ zs
            #lin = np.zeros((4, len(xs)))
            preds_lin = dat @ lin



            def opt_fun(abcdef_scaled):
                abcdef = abcdef_scaled * 1e-10
                new_preds = [new_pred_fun(x, y, abcdef) for x, y in zip(xs, ys)]
                errors = [(new_pred - pred)*1e10 for new_pred, pred in zip(new_preds, predictions)]
                error = sum(x**2 for x in errors)

                x = 2
                alpha = abcdef[0] * x + abcdef[1]
                beta = abcdef[2] * x + abcdef[3]

                alpha_min = abcdef[0]* 0 + abcdef[1]
                alpha_max = abcdef[0]* 3 + abcdef[1]
                r = lambda x: f'{x:.3e}'
                print('abcdef', [r(x) for x in abcdef_scaled],
                      '\tamin,amax', r(alpha_min), r(alpha_max), alpha_min < ys[0] < alpha_max,
                      'error', r(error))

                return error


            linear_slope = 2.1e-2 / 500e6
            linear_const = 1.465e-1 / 500e6 - 1.5e-10
            abcdef0 = [linear_slope, linear_const, linear_slope, linear_const*1.5, 0, 0]
            abcdef0_scaled = [x * 1e10 for x in abcdef0]

            import scipy
            opt_result = scipy.optimize.minimize(opt_fun, abcdef0_scaled)
            abcdef_opt_scaled = opt_result.x
            abcdef_opt = [x*1e-10 for x in abcdef_opt_scaled]

            preds_opt = [new_pred_fun(x, y, abcdef_opt) for x, y in zip(xs, ys)]
            preds_0 = [new_pred_fun(x, y, abcdef0) for x, y in zip(xs, ys)]





            # PLOTS

            #import matplotlib.pyplot as plt

            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(xs, ys, zs, color='b')
            #surf = ax.plot_trisurf(xs, ys, preds_opt, linewidth=0, alpha=0.5, color='r')
            ax.scatter(xs, ys, preds_opt, color='r')
            e_opt = sum(((zs-preds_opt)*1e10)**2)
            ax.set_title(f'Predictions with new model, {e_opt:.3e}')


            fig2 = plt.figure(3)
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.scatter(xs, ys, zs, color='b')
            ax2.scatter(xs, ys, preds_lin, color='g')
            e_lin = sum(((zs - preds_lin)*1e10)**2)
            ax2.set_title(f'Predictions with linear model, {e_lin:.3e}')

            fig2 = plt.figure(4)
            ax2 = fig2.add_subplot(111, projection='3d')
            ax2.scatter(xs, ys, zs, color='b')
            ax2.scatter(xs, ys, predictions, color='c')
            e_fix = sum(((zs - predictions)*1e10)**2)
            ax2.set_title(f'Predictions from fixture, {e_fix:.3e}')

            plt.show()

            #plt.scatter(xs, zs)
            #tempxs = list(range(17))
            ##tempys = [abcd_opt[0]*tempx + abcd_opt[1] for tempx in tempxs]
            ##tempys2 = [new_pred_fun(tempx, ys[0], abcd_opt) for tempx in tempxs]
            #tempys = [new_pred_fun(tempx, ys[0], abcdef_opt) for tempx in tempxs]
            #plt.plot(tempxs, tempys)
            ##plt.plot(tempxs, tempys2)
            #plt.show()



            #c = np.array(c)
            #plt.scatter(my_predictions, measured, c=c)
            #plt.grid()
            #plt.xlabel('Prediction based on input and fits')
            #plt.ylabel('Measured output')
            #plt.show()
            #c = np.array(c)
            #plt.scatter(predictions, measured, c=c)
            #plt.grid()
            #plt.xlabel('Prediction based on input and fits')
            #plt.ylabel('Measured output')
            #plt.show()
            return {}





    class GlitchTest(TemplateMaster.Test):
        analysis_outputs = ['first_glitch', 'last_glitch']
        parameters = ['glitch_start', 'glitch_end']
        parameter_algebra = {
            'first_glitch': 'glitch_start',
            'last_glitch': 'glitch_end',
        }
        NUM_DIVISIONS = 30

        def __init__(self, *args, **kwargs):
            self.saved_debug_image = False
            super().__init__(*args, **kwargs)

        def input_domain(self):
            return []

        def testbench(self, tester, values):
            freq = float(self.extras['frequency'])
            in_phase_delay = max(self.extras['phase_offset_range']) / freq

            # always between 0 and 1
            # rand_phase_offset = values[1]
            # "random" value within the specified range
            # phase_offset = offset_range[0] + rand_phase_offset*(offset_range[1]-offset_range[0])

            self.debug(tester, self.ports.in_a, 1 / freq * 100)
            self.debug(tester, self.ports.in_b, 1 / freq * 100)
            self.debug(tester, self.ports.out, 1 / freq * 100)
            # self.debug(tester, self.template.dut.thm_sel_bld[0], 1/freq*100)
            # self.debug(tester, self.template.dut.sel[0], 1/freq*100)

            #in_phase_delay = values['in_phase_delay']
            # always use maximum delay for glitch testing

            # the clock starts just after its falling edge
            tester.poke(self.ports.in_a, 0, delay={
                'freq': freq,
            })
            tester.delay(in_phase_delay)
            tester.poke(self.ports.in_b, 0, delay={
                'freq': freq,
            })
            tester.delay(1 / freq - in_phase_delay)


            ctrl_name = self.extras['control_input']
            ctrl = self.signals.from_circuit_name(ctrl_name)
            ctrl_min, ctrl_max = min(ctrl.value), max(ctrl.value)

            period = 1 / freq
            PERIODS = 5
            a_r, b_r, o_r, o_f, delay = [] , [] , [], [], []
            for i in range(self.NUM_DIVISIONS):

                # can't use fault clock or future pokes because the decimal ->
                # bits conversion is done in fixture
                input_delay = (i / self.NUM_DIVISIONS) * period
                tester.delay(input_delay)
                for j in range(PERIODS):
                    tester.poke(ctrl, ctrl_max)
                    tester.delay(period / 2)
                    tester.poke(ctrl, ctrl_min)
                    tester.delay(period / 2)
                tester.delay(period - input_delay)


                # only look at rising edges for now
                a_r.append(tester.get_value(self.ports.in_a, params={
                    'style': 'edge',
                    'forward': False,
                    'count': 2,
                    'rising': True
                }))
                b_r.append(tester.get_value(self.ports.in_b, params={
                    'style': 'edge',
                    'forward': False,
                    'count': 2,
                    'rising': True
                }))
                o_r.append(tester.get_value(self.ports.out, params={
                    'style': 'edge',
                    'forward': False,
                    'count': 5,
                    'rising': True,
                }))
                o_f.append(tester.get_value(self.ports.out, params={
                    'style': 'edge',
                    'forward': False,
                    'count': 5,
                    'rising': False,
                }))
                delay.append(input_delay)

            tester.poke(self.ports.in_a, 0)
            tester.poke(self.ports.in_b, 0)
            tester.delay(period)

            return [a_r, b_r, o_r, o_f, delay]

        def analysis(self, reads):
            freq = float(self.extras['frequency'])
            period = 1 / freq

            a_r, b_r, o_r, o_f, delay = reads


            xs = []
            ys1 = []
            ys2 = []
            ys3 = []
            ys4 = []
            gs = []
            for i, (a, b, r, f, d) in enumerate(zip(a_r, b_r, o_r, o_f, delay)):
                a, b, r, f = a.value, b.value, r.value, f.value
                time = a[-1]
                dist1 = r[2] - r[3]
                dist2 = r[1] - r[2]
                dist3 = f[2] - f[3]
                dist4 = f[1] - f[2]
                glitch = min(dist1, dist2, dist3, dist4) < 0.8 * period

                xs.append(d)
                ys1.append(dist1)
                ys2.append(dist2)
                ys3.append(dist3)
                ys4.append(dist4)
                gs.append(glitch)


            if not self.saved_debug_image:
                plt.plot(xs, ys1)
                plt.plot(xs, ys2)
                plt.plot(xs, ys3)
                plt.plot(xs, ys4)
                PlotHelper.save_current_plot('Glitch Test Debug')
                self.saved_debug_image = True


            if all(g == False for g in gs):
                g_start = 0
                g_end = 0
            elif all(g for g in gs):
                print('issue with glitch detection, why all glitches?')
                assert False
            else:
                # make start as soon as possible and end as late as possible
                # if there are multiple starts/ends
                # here I'm being lazy about what happens when the glitch wraps
                # around the end of a period to the start of the next
                N = len(gs)
                g_start = N+1
                g_end = -1
                for i in range(N):
                    current = gs[i]
                    next_ = gs[(i+1)%N]
                    if current == 0 and next_ == 1:
                        g_start = min(g_start, i)
                    elif current == 1 and next_ == 0:
                        g_end = max(g_end, i)

            # there's a +1 in the glitch_end calculation becauese we want the
            # first time there's no glitch; not the last time there is a glitch
            first_glitch = g_start / N
            last_glitch = (g_end+1) / N
            return {'first_glitch': first_glitch, 'last_glitch': last_glitch}

    tests = [
        #Test1,
        GlitchTest,
    ]


