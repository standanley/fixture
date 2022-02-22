from fault.domain_read import EdgeNotFoundError

from fixture.signals import SignalOut


class Representation:
    def __init__(self, name, params):
        self.name = name

        assert params['style'] in ['pulse_width', 'rising_edge_timer']
        self.style = params['style']
        self.reference_str = params['reference']
        self.reference = None

    def finish_init(self, signal_manager):
        if self.style == 'pulse_width':
            self.reference = signal_manager.from_circuit_name(self.reference_str)
            pulse_start = SignalOut(
                'real',
                f'{self.name}_pulse_start',
                None,
                None,
                True
            )
            ps_representation = Representation(f'{self.name}_pulse_start',
                                               {
                                                   'reference': None,
                                                   'style': 'rising_edge_timer',
                                               })
            ps_representation.reference = signal_manager.from_circuit_name('clk_v2t<0>')
            ps_representation.reference2 = self.reference
            pulse_start.representation = ps_representation
            signal_manager.add(pulse_start)
        elif self.style == 'rising_edge_timer':
            assert False, 'TODO'



    def representation_get_value(self, tester):
        if self.style == 'pulse_width':

            a = self.reference.spice_pin
            r_edge = tester.get_value(a,
                                    params={'style': 'edge', 'forward': False,
                                            'rising': True, 'count': 1})
            f_edge = tester.get_value(a,
                                    params={'style': 'edge', 'forward': False,
                                            'rising': False, 'count': 1})

            def callback():
                try:
                    r_time = r_edge.value[0]
                    f_time = f_edge.value[0]
                    return float(f_time - r_time)
                except EdgeNotFoundError:
                    return 0

            return tester.GetValueReturnObject(callback)
        elif self.style == 'rising_edge_timer':
            r1 = self.reference.spice_pin
            r2 = self.reference2.spice_pin
            r1_edge = tester.get_value(r1,
                                      params={'style': 'edge', 'forward': False,
                                              'rising': True, 'count': 1})
            r2_edge = tester.get_value(r2,
                                      params={'style': 'edge', 'forward': False,
                                              'rising': True, 'count': 1})

            def callback():
                try:
                    r1_time = r1_edge.value[0]
                    r2_time = r2_edge.value[0]
                    return float(r2_time - r1_time)
                except EdgeNotFoundError:
                    return 0

            return tester.GetValueReturnObject(callback)
        else:
            assert False, f'unknown proxy style {self.style}'

