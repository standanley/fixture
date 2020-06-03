from fixture import TemplateMaster
from fixture import RealIn
import fault
from magma import *
import pytest


class SimpleBufTemplate(TemplateMaster):
    required_ports = ['in_single', 'out_single']

    class Test1(TemplateMaster.Test):
        parameter_algebra = {}
        def input_domain(self):
            return []
        def testbench(self, tester, values):
            return []
        def analysis(self, reads):
            return {}

    tests = []

#    @classmethod
#    def specify_test_inputs(self):
#        return []
#    @classmethod
#    def specify_test_outputs(self):
#        return []
#    @classmethod
#    def run_single_test(self, tester, value):
#        pass
#    @classmethod
#    def process_single_test(self, tester):
#        return []

class SimpleBufCirc(Circuit):
    __name__ = 'mybuf'
    io = IO(myin = BitIn,
            myout = BitOut)

mapping = {'in_single': 'myin', 'out_single': 'myout'}

def test_require_required_ports():
    with pytest.raises(AssertionError):
        class BadAmpTemplate(TemplateMaster):
            name = 'forgot to add required_ports'

        t = BadAmpTemplate(SimpleBufCirc, {}, None)

def test_require_good_mapping():
    with pytest.raises(AssertionError):
        t = SimpleBufTemplate(SimpleBufCirc, {'in_single': 'myin'}, None)

def test_required_port_info():
    t = SimpleBufTemplate(SimpleBufCirc, mapping, None)
    text = t.required_port_info()
    # text should at least mention each port
    for port_name in SimpleBufTemplate.required_ports:
        assert port_name in text

def test_port_sorting():

    t = SimpleBufTemplate(SimpleBufCirc,
        {'in_single': 'myin',
        'out_single': 'myout'},
        None
    )
    # TODO actually check sorting on a more complicated example


# TODO maybe do a full test on a circuit defined here

if __name__ == '__main__':
    #test_simple()
    test_port_sorting()
