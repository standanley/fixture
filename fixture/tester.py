import fault
from fault.domain_read import EdgeNotFoundError

from fixture.signals import SignalOut


class Tester(fault.Tester):

    #def poke(self, port, value, delay=None):
    #    if isinstance(port, SignalIn):
    #        # TODO in the future we should take only signals as arg
    #        assert hasattr(port, 'representation')
    #        r = port.representation
    #
    #    else:
    #        return super().poke(port, value, delay=delay)

    class GetValueReturnObject:
        def __init__(self, callback):
            self.callback = callback

        def __getattr__(self, item):
            if item == 'value':
                return self.callback()
            raise AttributeError

    def get_value(self, port, params=None):
        if isinstance(port, SignalOut):
            # TODO in the future we should take only signals as arg
            if hasattr(port, 'representation'):
                return port.representation.representation_get_value(self)
            else:
                magma_port = port.spice_pin
                return super().get_value(magma_port, params=params)

        else:
            return super().get_value(port, params=params)