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
                r = port.representation
                if r['style'] == 'pulse_width':

                    a = r['reference'].spice_pin
                    r_edge = self.get_value(a, params={'style': 'edge', 'forward':False, 'rising':True, 'count':1})
                    f_edge = self.get_value(a, params={'style': 'edge', 'forward':False, 'rising':False, 'count':1})

                    def callback():
                        try:
                            r_time = r_edge.value[0]
                            f_time = f_edge.value[0]
                            return float(f_time - r_time)
                        except EdgeNotFoundError:
                            return 0

                    return self.GetValueReturnObject(callback)
                else:
                    assert False, f'unknown proxy style {r["style"]}'
            else:
                magma_port = port.spice_pin
                return super().get_value(magma_port, params=params)

        else:
            return super().get_value(port, params=params)