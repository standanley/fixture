from fault.domain_read import EdgeNotFoundError, domain_read
from fixture.signals import SignalOut, SignalIn, SignalArray, Signal
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

    @staticmethod
    def convert_and_tag_referenced_signals(params, signals):
        def get_and_tag(name):
            # TODO might be bad if ref_signal is SA, but also I think we are phasing out is_proxy_component
            try:
                s = signals.from_circuit_name(name)
            except KeyError:
                raise KeyError(f'Could not find signal "{name}" when creating a proxy signal')
            s.is_proxy_component = True
            return s
        if params['style'] == 'pulse_width':
            params['reference'] = get_and_tag(params['reference'])
        elif (params['style'] == 'linear_combination_in'
              or params['style'] == 'linear_combination_out'
              or params['style'] == 'vector'):
            params['components'] = [get_and_tag(n) for n in params['components']]
        else:
            assert False, f'Unknown proxy style {params["style"]}'


    @classmethod
    def create_signal(cls, c_name, t_name, params):
        if params['style'] == 'pulse_width':

            s = SignalOut(
                'real',
                c_name,
                None,
                t_name,
                t_name is None,
                None
            )
        elif params['style'] == 'vector':
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
            #class PlaceholderSignal:
            #    spice_name = None
            #    template_name = None
            #    representation = None
            s = SignalArray(
                #np.zeros(len(params['components']), dtype=object),
                np.array(params['components']),
                None,
                template_name=t_name,
                spice_name=c_name
            )
        elif params['style'] == 'linear_combination_in':
            s = SignalIn(
                params.get('value', None),
                'will_complete_in_finish_init',
                'will_complete_in_finish_init',
                'will_complete_in_finish_init',
                'will_complete_in_finish_init',
                c_name,
                None,
                t_name,
                'will_complete_in_finish_init',
                'will_complete_in_finish_init',
            )
        elif params['style'] == 'linear_combination_out':
            # TODO I think bus_info maybe should not be None
            s = SignalOut(
                'will_complete_in_finish_init',
                c_name,
                None,
                t_name,
                'will_complete_in_finish_init',
                None
            )
        else:
            assert False, f'Unknown proxy style {params["style"]}'
        rep = cls(c_name, s, params)
        s.representation = rep
        return s

    def __init__(self, name, parent_signal, params):
        self.name = name

        assert params['style'] in ['pulse_width', 'rising_edge_timer',
                                   'linear_combination_in',
                                   'linear_combination_out', 'vector']
        self.style = params['style']
        self.params = params
        self.parent_signal = parent_signal

    def finish_init(self, signal_manager):
        if self.style == 'pulse_width':
            assert isinstance(self.params['reference'], Signal)
            ## overwrite string version of reference with actual object
            #self.params['reference_str'] = self.params['reference']
            #self.params['reference'] = signal_manager.from_circuit_name(
            #    self.params['reference_str'])
            pulse_start = SignalOut(
                'real',
                f'{self.name}_pulse_start',
                None,
                None,
                True,
                None
            )
            # TODO do I put self.parent_signal as the parent signal for
            # the pulse_start too?
            ps_representation = Representation(f'{self.name}_pulse_start',
                                               None,
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
        elif self.style == 'linear_combination_in' or self.style == 'linear_combination_out':
            # with new config parse, components have already been converted
            #components_str = []
            #components = []
            #for component_str in self.params['components']:
            #    components_str.append(component_str)
            #    component = signal_manager.from_circuit_name(component_str)
            #    components.append(component)
            #self.params['components_str'] = components_str
            #self.params['components'] = components

            # copy signal properties from one of the components
            ref_signal = self.params['components'][0]
            if self.style == 'linear_combination_in':
                # TODO does it makes sense to calculate the value from refs?
                # I think not, but I don't know the right way to handle this
                assert isinstance(ref_signal, SignalIn)
                self.parent_signal.is_true_digital = ref_signal.is_true_digital
                self.parent_signal.get_random = ref_signal.get_random
                self.parent_signal.auto_set = ref_signal.auto_set
                self.parent_signal.optional_expr = ref_signal.optional_expr
                val = self.parent_signal.value
                nominal = val[0] if val is not None and len(val) == 1 else None
                self.parent_signal.nominal = nominal
            elif self.style == 'linear_combination_out':
                assert isinstance(ref_signal, SignalOut)
                self.parent_signal.type_ = ref_signal.type_
                self.parent_signal.auto_measure = ref_signal.auto_measure
            else:
                assert False, f'TODO make linear combination out of signal type {type(ref_signal)}'
        elif self.style == 'vector':
            # with new config parse, components have already been converted
            ## convert component strings to actual objects
            #components_str = []
            #components = []
            #for component_str in self.params['components']:
            #    components_str.append(component_str)
            #    component = signal_manager.from_circuit_name(component_str)
            #    components.append(component)
            #self.params['components_str'] = components_str
            #self.params['components'] = components
            #self.parent_signal.array = np.array(components)

            if self.parent_signal.template_name is not None:
                for i, s in enumerate(self.parent_signal.array):
                    # TODO should this be recursive?
                    if s.template_name is None:
                        s.template_name = f'{self.parent_signal.template_name}[{i}]'




    def representation_get_value(self, tester, params=None):
        if params == None:
            params = {}
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

        elif self.style == 'linear_combination_in' or self.style == 'linear_combination_out':
            components = self.params['components']
            gvs = [tester.get_value(s.spice_pin, params=params) for s in components]
            coefs = self.params['coefficients']
            def callback():
                if (len(gvs)>0 and gvs[0].params.get('style', None) == 'block'):
                    # probably block read
                    ts, vs = zip(*[gv.value for gv in gvs])
                    time_check = all([np.array_equal(ts[0], ts[i])
                                      for i in range(1, len(ts))])
                    assert time_check, f'Inconsistent time steps in block read of vector {self.name}'
                    t = ts[0]
                    v = sum(vi*coef for vi, coef in zip(vs, coefs))
                    return t, v

                else:
                    return sum(gv.value*coef for gv, coef in zip(gvs, coefs))
            return tester.GetValueReturnObject(callback)
        else:
            assert False, f'unknown proxy style {self.style}'

