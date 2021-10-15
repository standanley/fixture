from fixture import TemplateMaster
from fixture import signals
import fault
from magma import *
import pytest

from fixture.signals import SignalManager


def get_test_signal(template_name, spice_name):
    return signals.SignalIn(
        None,
        'analog',
        False,
        False,
        spice_name,
        None,
        template_name,
        False
    )


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

def test_require_required_ports():
    with pytest.raises(AssertionError):
        class BadAmpTemplate(TemplateMaster):
            name = 'forgot to add required_ports'

        t = BadAmpTemplate(SimpleBufCirc, None, [])

def test_require_good_mapping():
    with pytest.raises(AssertionError):
        signals = [get_test_signal('in_single', 'myin')]
        mapping = {'in_single': signals[0]}
        sm = SignalManager(signals, mapping)
        t = SimpleBufTemplate(SimpleBufCirc, None, sm)

def test_required_port_info():
    signals = [
        get_test_signal('in_single', 'myin'),
        get_test_signal('out_single', 'myout')
    ]
    mapping = {'in_single': signals[0], 'out_single': signals[1]}
    sm = SignalManager(signals, mapping)
    t = SimpleBufTemplate(SimpleBufCirc, None, sm)
    text = t.required_port_info()
    # text should at least mention each port
    for port_name in SimpleBufTemplate.required_ports:
        assert port_name in text

# TODO maybe do a full test on a circuit defined here

if __name__ == '__main__':
    #test_required_port_info()
    test_require_good_mapping()
