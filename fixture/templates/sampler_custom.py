from fixture.templates import SamplerTemplate
import random


class SamplerCustomTemplate(SamplerTemplate):


    def __init__(self, *args, **kwargs):
        #self.required_ports.append('ignore')
        super().__init__(*args, **kwargs)
        pass


    def read_value(self, tester, port, wait):
        r = tester.get_value(port, params={'style': 'edge', 'forward':False, 'rising':False, 'count':1, 'on_error': [-1]})
        r2 = tester.get_value(port, params={'style': 'edge', 'forward':False, 'rising':True, 'count':1, 'on_error': [-1]})
        assert wait == 0
        tester.delay(wait)
        return r, r2

    def interpret_value(self, read):
        r, r2 = read
        return r.value[0] - r2.value[0]



