import fault
from fault.domain_read import EdgeNotFoundError

from fixture.signals import SignalIn, SignalOut, SignalArray
import numpy as np


class Tester(fault.Tester):

    def poke(self, port, value, delay=None):
        if isinstance(port, list):
            # TODO can we poke digital port with integer and assume binary?
            assert len(value) == len(port)
            for p, v in zip(port, value):
                # TODO an old bug in domain_read caused issues here because
                # we passed the same delay object in each time instead of a copy
                self.poke(p, v, delay=delay)
            # TODO this has no return value ... I think
            return

        if isinstance(port, SignalIn):
            ## TODO in the future we should take only signals as arg
            #assert hasattr(port, 'representation')
            #r = port.representation
            return self.poke(port.spice_pin, value, delay=delay)

        elif isinstance(port, SignalArray):
            def is_lc(s):
                return (s.representation is not None
                        and s.representation.style == 'linear_combination')
            if any(is_lc(s) for s in port):
                # linear combination
                assert all(is_lc(s) for s in port), 'TODO mixed lc and not in single poke'
                #rows = []
                columns = set()
                for s in port:
                    #rows.append(s)
                    for component in s.representation.params['components']:
                        columns.add(component)
                columns = list(columns)
                matrix = np.zeros((len(list(port)), len(columns)))
                for i, s in enumerate(port):
                    params = s.representation.params
                    for component, coef in zip(params['components'], params['coefficients']):
                        matrix[i][columns.index(component)] = coef

                try:
                    matrix_inv = np.linalg.inv(matrix)
                    transformed_value = matrix_inv @ np.array(value)
                    return self.poke(columns, transformed_value)

                except np.linalg.LinAlgError as e:
                    raise np.linalg.LinAlgError(f'Error converting domains for {port}: {e}')

            else:
                # not a linear combination, so just match port/value one-to-one
                assert len(port.array) == len(value)
                return self.poke(list(port), value, delay=delay)

        else:
            return super().poke(port, value, delay=delay)

    class GetValueReturnObject:
        def __init__(self, callback):
            self.callback = callback

        def __getattr__(self, item):
            if item == 'value':
                return self.callback()
            raise AttributeError

    def get_value(self, port, params=None):
        if isinstance(port, (SignalIn, SignalOut)):
            # TODO in the future we should take only signals as arg
            if hasattr(port, 'representation') and port.representation is not None:
                return port.representation.representation_get_value(self)
            else:
                magma_port = port.spice_pin
                return super().get_value(magma_port, params=params)

        elif isinstance(port, SignalArray):
            get_value_objects = []
            for sub_array in port:
                gvo = self.get_value(sub_array, params=params)
                get_value_objects.append(gvo)
            return self.GetValueReturnObject(
                lambda: [gvo.value for gvo in get_value_objects]
            )
        else:
            return super().get_value(port, params=params)