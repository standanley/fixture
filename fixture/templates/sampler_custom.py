from fixture.templates import SamplerTemplate
import random


class SamplerCustomTemplate(SamplerTemplate):


    def __init__(self, *args, **kwargs):
        #self.required_ports.append('ignore')
        super().__init__(*args, **kwargs)
        pass


    def read_value(self, tester, port, wait):
        #return 4242
        r = tester.get_value(port, params={'style': 'edge', 'forward':True, 'rising':False, 'count':1})
        r2 = tester.get_value(port, params={'style': 'edge', 'forward':True, 'rising':True, 'count':1})
        tester.delay(wait)
        return r, r2

    def interpret_value(self, read):
        #return random.random()
        r, r2 = read
        return r.value[0] - r2.value[0]


    def schedule_clk(self, tester, port, value, wait):
        clks = self.extras['clks']
        main = self.get_name_circuit(port.name.array)
        unit = float(clks['unit'])
        period = clks['period']
        clks = {k:v for k,v in clks.items() if (k!='unit' and k!='period')}
        num_samplers = getattr(self.dut, main).N
        desired_sampler = port.name.index


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
                for p in range(played_periods):
                    for time, val in clks[clk].items():
                        time_transform = time + (period - period_start_time)
                        if time_transform > period:
                            time_transform -= period
                        #time_transform *= unit
                        time_transform_shift = time_transform + (p - measured_period) * period
                        temp.append((time_transform_shift, val))
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
                tester.poke(getattr(self.dut, clk)[i], 0, delay={
                    'type': 'future',
                    'waits': waits,
                    'values': values
                })


