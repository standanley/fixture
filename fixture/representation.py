from fault.domain_read import EdgeNotFoundError

from fixture.signals import SignalOut, SignalIn, SignalArray
import numpy as np


class Representation:

    #class VectorSignalIn(SignalIn):
    #    def __setattr__(self, key, value):
    #        # we only want to do this special thing after the init finishes
    #        # because the representation won't be set yet otherwise
    #        if key == 'get_random' and hasattr(self, 'get_random'):
    #            # bubble down?
    #            for s in self.representation.params['components']:
    #                s.__setattr__('get_random', value)
    #        else:
    #            super().__setattr__(key, value)

    @classmethod
    def create_signal(cls, c_name, t_name, params):
        if params['style'] == 'pulse_width':

            s = SignalOut(
                'real',
                c_name,
                None,
                t_name,
                t_name is None
            )
        if params['style'] == 'vector':
            #s = cls.VectorSignalIn(
            #    None,
            #    'real',
            #    False,
            #    False,
            #    c_name,
            #    None,
            #    t_name,
            #    None,
            #    representation=rep
            #)
            class PlaceholderSignal:
                spice_name = None
                template_name = None
                representation = None
            s = SignalArray(
                #np.zeros(len(params['components']), dtype=object),
                np.array([PlaceholderSignal()]*len(params['components'])),
                {},
                template_name=t_name,
                spice_name=c_name
            )
        rep = cls(c_name, s, params)
        s.representation = rep
        return s

    def __init__(self, name, parent_signal, params):
        self.name = name

        assert params['style'] in ['pulse_width', 'rising_edge_timer',
                                   'linear_combination', 'vector']
        self.style = params['style']
        self.params = params
        self.parent_signal = parent_signal

    def finish_init(self, signal_manager):
        if self.style == 'pulse_width':
            # overwrite string version of reference with actual object
            self.params['reference_str'] = self.params['reference']
            self.params['reference'] = signal_manager.from_circuit_name(
                self.params['reference_str'])
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
            ps_representation.params['reference'] = signal_manager.from_circuit_name('clk_v2t<0>')
            ps_representation.params['reference2'] = self.params['reference']
            pulse_start.representation = ps_representation
            signal_manager.add(pulse_start)
        elif self.style == 'rising_edge_timer':
            assert False, 'TODO'
        elif self.style == 'linear_combination':
            assert False, 'TODO'
        elif self.style == 'vector':
            # convert component strings to actual objects
            components_str = []
            components = []
            for component_str in self.params['components']:
                components_str.append(component_str)
                component = signal_manager.from_circuit_name(component_str)
                components.append(component)
            self.params['components_str'] = components_str
            self.params['components'] = components
            self.parent_signal.array = np.array(components)

            if self.parent_signal.template_name is not None:
                for i, s in enumerate(self.parent_signal.array):
                    # TODO should this be recursive?
                    if s.template_name is None:
                        s.template_name = f'{self.parent_signal.template_name}[{i}]'




    def representation_get_value(self, tester):
        if self.style == 'pulse_width':

            a = self.params['reference'].spice_pin
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
            r1 = self.params['reference'].spice_pin
            r2 = self.params['reference2'].spice_pin
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

        elif self.style == 'vector':
            components = self.params['components']
            gvs = [tester.get_value(s.spice_pin) for s in components]
            def callback():
                return [gv.value for gv in gvs]
            return tester.GetValueReturnObject(callback)

        else:
            assert False, f'unknown proxy style {self.style}'

