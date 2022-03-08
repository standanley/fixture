import fault
from fault.domain_read import EdgeNotFoundError

from fixture.signals import SignalIn, SignalOut, SignalArray


class Tester(fault.Tester):

    def poke(self, port, value, delay=None):
        if isinstance(port, SignalIn):
            ## TODO in the future we should take only signals as arg
            #assert hasattr(port, 'representation')
            #r = port.representation
            return self.poke(port.spice_pin, value, delay=delay)

        elif isinstance(port, SignalArray):
            assert len(port.array) == len(value)
            for p, v in zip(port, value):
                # TODO an old bug in domain_read caused issues here because
                # we passed the same delay object in each time instead of a copy
                self.poke(p, v, delay=delay)
            # TODO this has no return value ... I think
            return

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