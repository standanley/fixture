from fixture import *
import fixture.real_types as rt
import fault
import magma

def test_bit():
    a = rt.bit
    a = a(None)
    a = rt.input(a)
    assert rt.is_bit(a)
    assert not rt.is_real(a)
    assert not rt.is_binary_analog(a)
    assert a.is_input()
    assert not a.is_output()

    b = rt.bit
    b = b(None)
    b = rt.output(b)
    assert rt.is_bit(b)
    assert not rt.is_real(b)
    assert not rt.is_binary_analog(b)
    assert not b.is_input()
    assert b.is_output()

def test_real_friendly():
    a = RealIn((1, 5), 'a')
    assert not rt.is_bit(a)
    assert rt.is_real(a)
    assert not rt.is_binary_analog(a)
    assert a.limits == (1, 5)
    assert a.name == 'a'
    assert a.is_input()
    assert not a.is_output()

    b = RealOut((1, 5), 'b')
    assert not rt.is_bit(b)
    assert rt.is_real(b)
    assert not rt.is_binary_analog(b)
    assert b.limits == (1, 5)
    assert b.name == 'b'
    assert not b.is_input()
    assert b.is_output()

def test_real():
    a = rt.real
    a = a((1, 5))
    a = rt.input(a)

    assert not rt.is_bit(a)
    assert rt.is_real(a)
    assert not rt.is_binary_analog(a)
    assert a.limits == (1, 5)
    assert a.is_input()
    assert not a.is_output()

    b = rt.real
    b = b(3.3)
    b = rt.output(b)
    assert not rt.is_bit(b)
    assert rt.is_real(b)
    assert not rt.is_binary_analog(b)
    assert b.limits == 3.3
    assert not b.is_input()
    assert b.is_output()

    f = fault.RealIn
    assert rt.is_real(f)

def test_binary_analog_friendly():
    a = BinaryAnalogIn(None, 'a')
    assert not rt.is_bit(a)
    assert not rt.is_real(a)
    assert rt.is_binary_analog(a)
    assert a.name == 'a'
    assert a.is_input()
    assert not a.is_output()

    b = BinaryAnalogIn()
    assert not rt.is_bit(b)
    assert not rt.is_real(b)
    assert rt.is_binary_analog(b)
    assert a.is_input()
    assert not a.is_output()

    try:
        bad = BinaryAnalogIn((1, 5), 'bad')
        assert False, 'Passing value to BA should fail'
    except AssertionError:
        pass

def test_binary_analog():

    test = magma.Bit
    test2 = rt.input(test)
    print(test)
    print(test2)


    a = rt.binary_analog
    a = a(None)
    a = rt.input(a)
    assert not rt.is_bit(a)
    assert not rt.is_real(a)
    assert rt.is_binary_analog(a)
    assert a.is_input()
    assert not a.is_output()

    # Although it works, BA output doesn't really make sense
    # b = rt.binary_analog
    # b = b(None)
    # b = rt.output(b)
    # assert not rt.is_bit(b)
    # assert not rt.is_real(b)
    # assert rt.is_binary_analog(b)
    # assert not b.is_input()
    # assert b.is_output()

def test_magma_interaction():

    class TestCircuit(magma.Circuit):
        io = magma.IO(
            mag_b_in  = magma.BitIn,
            mag_b_out = magma.BitOut,
            fix_b_in  = rt.input(rt.bit(None)),
            fix_b_out = rt.output(rt.bit(None)),

            fau_r_in  = fault.RealIn,
            fau_r_out = fault.RealOut,
            fix_r_in  = RealIn((0, 5)),
            fix_r_out = RealOut((0, 3.3)),

            fix_ba_in = BinaryAnalogIn()
        )

    for name, port in TestCircuit.io.ports.items():
        # note that input/output gets flipped in magma circuits
        if '_in' in name:
            assert not port.is_input()
        if '_out' in name:
            assert not port.is_output()

        is_bit = '_b_' in name
        is_real = '_r_' in name
        is_ba = '_ba_' in name
        assert not is_bit ^ rt.is_bit(port)
        assert not is_real ^ rt.is_real(port)
        assert not is_ba ^ rt.is_binary_analog(port)
        print(name, isinstance(port, magma.Type))

    for name, port in TestCircuit.IO.ports.items():
        print(name, isinstance(port, magma.Type))


    ports = TestCircuit.io.ports


if __name__ == '__main__':
    test_magma_interaction()
    #test_real()
    #test_binary_analog_friendly()
    #test_binary_analog()
    #test_bit()

