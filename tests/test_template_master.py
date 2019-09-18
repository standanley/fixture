from facet import *
from magma import *
import pytest


class SimpleBufTemplate(TemplateMaster):
    __name__ = 'abc123'
    required_ports = ['in_single', 'out_single']

def test_require_required_ports():
    with pytest.raises(AssertionError):
        class BadAmpTemplate(TemplateMaster):
            name = 'forgot to add required_ports'

def test_require_mapping():
    with pytest.raises(AssertionError):
        class BadInterface(SimpleBufTemplate):
            IO = ['my_a', In(Bit)]
            name = 'Forgot to define "mapping" method'

def test_require_good_mapping():
    with pytest.raises(AssertionError):
        class BadInterface(SimpleBufTemplate):
            IO = ['my_a', In(Bit)]
            def mapping(self):
                self.in_single = self.my_a
                # forgot to assign out_single

def test_required_port_info():
    text = SimpleBufTemplate.required_port_info()
    # text should at least mention each port
    for port_name in SimpleBufTemplate.required_ports:
        assert port_name in text

def test_simple():
    class UserBufInterface(SimpleBufTemplate):
        name = 'MyBufInterface'
        IO = ['my_in', In(Bit), 'my_out', Out(Bit)]

        def mapping(self):
            self.in_single = self.my_in
            self.out_single = self.my_out

    class MyBuf1(UserBufInterface):
        @classmethod
        def definition(io):
            io.my_out <= io.my_in

    def test_thing():
        t = fault.Tester(MyBuf1)

        # first with user-defined names
        t.poke(MyBuf1.my_in, 0)
        t.eval()
        t.expect(MyBuf1.my_out, 0)

        t.poke(MyBuf1.my_in, 1)
        t.eval()
        t.expect(MyBuf1.my_out, 1)

        # now with template names
        t.poke(MyBuf1.in_single, 0)
        t.eval()
        t.expect(MyBuf1.out_single, 0)

        t.poke(MyBuf1.in_single, 1)
        t.eval()
        t.expect(MyBuf1.out_single, 1)

        t.compile_and_run('verilator')

