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
    assert rt.is_input(a)
    assert not rt.is_output(a)

    b = rt.bit
    b = b(None)
    b = rt.output(b)
    assert rt.is_bit(b)
    assert not rt.is_real(b)
    assert not rt.is_binary_analog(b)
    assert not b.is_input()
    assert b.is_output()
    assert not rt.is_input(b)
    assert rt.is_output(b)

def test_real_friendly():
    a = RealIn((1, 5), 'a')
    assert not rt.is_bit(a)
    assert rt.is_real(a)
    assert not rt.is_binary_analog(a)
    assert a.limits == (1, 5)
    assert a.name == 'a'
    assert rt.get_name(a) == 'a'
    assert a.is_input()
    assert not a.is_output()
    assert rt.is_input(a)
    assert not rt.is_output(a)

    b = RealOut((1, 5), 'b')
    assert not rt.is_bit(b)
    assert rt.is_real(b)
    assert not rt.is_binary_analog(b)
    assert b.limits == (1, 5)
    assert b.name == 'b'
    assert rt.get_name(b) == 'b'
    assert not b.is_input()
    assert b.is_output()
    assert not rt.is_input(b)
    assert rt.is_output(b)

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
    assert rt.is_input(a)
    assert not rt.is_output(a)

    b = rt.real
    b = b(3.3)
    b = rt.output(b)
    assert not rt.is_bit(b)
    assert rt.is_real(b)
    assert not rt.is_binary_analog(b)
    assert b.limits == 3.3
    assert not b.is_input()
    assert b.is_output()
    assert not rt.is_input(b)
    assert rt.is_output(b)

    f = fault.RealIn
    assert rt.is_real(f)

def test_binary_analog_friendly():
    a = BinaryAnalogIn(None, 'a')
    assert not rt.is_bit(a)
    assert not rt.is_real(a)
    assert rt.is_binary_analog(a)
    assert a.name == 'a'
    assert rt.get_name(a) == 'a'
    assert a.is_input()
    assert not a.is_output()
    assert rt.is_input(a)
    assert not rt.is_output(a)

    b = BinaryAnalogIn()
    assert not rt.is_bit(b)
    assert not rt.is_real(b)
    assert rt.is_binary_analog(b)
    assert b.is_input()
    assert not b.is_output()
    assert rt.is_input(b)
    assert not rt.is_output(b)

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
    assert rt.is_input(a)
    assert not rt.is_output(a)

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

            fix_ba_in = BinaryAnalogIn(),

            fix_arr_b_in = rt.Array(1, magma.BitIn),
            fix_arr_b_out = rt.Array(1, magma.BitOut),
            fix_arr_r_in  = rt.Array(5, rt.RealIn(0, 1)),
            fix_arr_r_out = rt.Array(3, rt.RealOut(0, 1)),
            fix_arr_ba_in = rt.Array(1, rt.BinaryAnalogIn())
        )

    for name, port in TestCircuit.io.ports.items():
        # note that input/output gets flipped in magma circuits
        if '_in' in name:
            assert rt.is_input(port)
            assert not port.is_input()
        if '_out' in name:
            assert rt.is_output(port)
            assert not port.is_output()


        is_bit = '_b_' in name
        is_real = '_r_' in name
        is_ba = '_ba_' in name
        assert not is_bit ^ rt.is_bit(port)
        assert not is_real ^ rt.is_real(port)
        assert not is_ba ^ rt.is_binary_analog(port)

        is_array = '_arr_' in name
        assert not is_array ^ rt.is_array(port)

        assert rt.get_name(port) == name
        #print(name, isinstance(port, magma.Type))


    # BONUS: I don't think accessing TestCircuit.IO is recommended because
    # it is just literally what was passed in during declaration, but that's
    # handy for us because it lets us test all these types before instantiation
    for name, port in TestCircuit.IO.ports.items():
        #print(name, isinstance(port, magma.Type))
        if '_in' in name:
            assert rt.is_input(port)
            assert port.is_input()
        if '_out' in name:
            assert rt.is_output(port)
            assert port.is_output()

        is_bit = '_b_' in name
        is_real = '_r_' in name
        is_ba = '_ba_' in name
        assert not is_bit ^ rt.is_bit(port)
        assert not is_real ^ rt.is_real(port)
        assert not is_ba ^ rt.is_binary_analog(port)

        is_array = '_arr_' in name
        assert not is_array ^ rt.is_array(port)

def test_array():
    ar = rt.Array(3, rt.RealIn((0, 1)))
    aba = rt.Array(3, rt.BinaryAnalogIn())

    assert not rt.is_bit(ar)
    assert rt.is_real(ar)
    assert not rt.is_binary_analog(ar)

    assert not rt.is_bit(aba)
    assert not rt.is_real(aba)
    assert rt.is_binary_analog(aba)

if __name__ == '__main__':
    test_magma_interaction()
    #test_real()
    #test_binary_analog_friendly()
    #test_binary_analog()
    #test_bit()
    #test_array()

