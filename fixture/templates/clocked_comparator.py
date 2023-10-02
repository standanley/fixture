from fixture import TemplateMaster
from fixture.signals import create_input_domain_signal
from fixture.template_creation_utils import extract_pzs


class ClockedComparatorTemplate(TemplateMaster):
    required_ports = ['in_pos', 'in_neg', 'out_pos', 'out_neg', 'clk']
    required_info = {
        'approx_settling_time': 'Approximate time it takes for comparator to settle from a 1% input difference',
        'common_mode_nominal': 'Will be used as common mode for all tests (for now)',
        'differential_max': 'Maximum input signal, defined as (in_pos - in_neg)',
        'time_for_growth_measurement': 'When measuring exponential growth, look at this many seconds after the clock raises'
    }

    class TimingTest(TemplateMaster.Test):
        analysis_outputs = ['growth_constant']
        parameters = [
            'growth_constant0'
        ]
        parameter_algebra = {
            'growth_constant': 'growth_constant0',
        }
        bounds_dict = {
        }
        vector_mapping = {}

        def input_domain(self):
            maximum = self.extras['differential_max']
            percentage = 0.01
            return [create_input_domain_signal('in_start', (-maximum*percentage, maximum*percentage))]

        def testbench(self, tester, values):
            debug_time = self.extras['approx_settling_time']*150
            self.debug(tester, self.signals.in_pos, debug_time)
            self.debug(tester, self.signals.in_neg, debug_time)
            self.debug(tester, self.signals.out_pos, debug_time)
            self.debug(tester, self.signals.out_neg, debug_time)
            self.debug(tester, self.signals.clk, debug_time)

            diff = values['in_start']
            cm = self.extras['common_mode_nominal']
            pos = cm + diff/2
            neg = cm - diff/2

            tester.poke(self.signals.clk, 0)
            tester.poke(self.signals.in_pos, pos)
            tester.poke(self.signals.in_neg, neg)
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)


            growth_time = self.extras['time_for_growth_measurement']
            read_params = {'style': 'block', 'duration': growth_time}
            pos_wave = tester.get_value(self.signals.out_pos, read_params)
            neg_wave = tester.get_value(self.signals.out_neg, read_params)
            tester.poke(self.signals.clk, 1)
            tester.delay(wait_time)

            return [pos_wave, neg_wave]

        def analysis(self, reads):
            #
            # if isinstance(self.signals.input, SignalArray):
            #     # we are vectored

            pos_wave, neg_wave = (read.value for read in reads)
            assert all(pos_wave[0] == neg_wave[0]), 'TODO differential signal for different pos and neg timesteps'
            diff_vals = pos_wave[1] - neg_wave[1]
            NP, NZ = 1, 0
            ps, zs = extract_pzs(NP, NZ, pos_wave[0], diff_vals)

            growth_constant = ps[0]



            return {'growth_constant': growth_constant}

        def post_regression(self, regression_models, regression_dataframe):
            return {}

    tests_all = [TimingTest]
    tests_default = [TimingTest]
