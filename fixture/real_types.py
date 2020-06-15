import fault
import magma

def Real(limits=None, name=None):
    class FixtureRealType(fault.ms_types.RealType):
        pass
    FixtureRealType.name = name
    FixtureRealType.limits = limits
    return FixtureRealType

def RealIn(limits, name=None):
    return Real(limits, name)[magma.Direction.In]
    
def RealOut(limits=None, name=None):
    return Real(limits, name)[magma.Direction.Out]


'''
The Binary Analog type accepts a value as input to be consistent with the 
Real type, but that value must be None (or unspecified) because it does
not make sense to have a BA port with a pinned value.
'''

class BinaryAnalogType(magma.Bit):
    # This is analogous to fault.RealType
    pass

def BinaryAnalog(value=None, name=None):
    assert value is None, 'Binary analog port can not have a value'
    class FixtureBaType(BinaryAnalogType):
        pass
    FixtureBaType.name = name
    return FixtureBaType

def BinaryAnalogIn(value=None, name=None):
    return BinaryAnalog(value, name)[magma.Direction.In]

# BA output does not really make sense
#def BinaryAnalogOut(value=None, name=None):
#    return BinaryAnalogType(value, name)[magma.Direction.Out]


'''
Like the BA type, we create a Bit type that accepts limits for consistency.
If the user wants to use BitIn syntax they can use magma.BitIn
'''
def Bit(limits=None):
    assert limits==None, 'Bit type cannot have limits'
    return magma.Bit

def Array(n, t):
    return magma.Array[n, t]


'''
Various tools for sorting ports
'''
def get_type(x):
    y = type(x) if isinstance(x, magma.Type) else x
    z = y.T if issubclass(y, magma.Array) else y
    return z

def is_real(x):
    t = get_type(x)
    # we will allow this to accept fault versions as well as fixture versions
    return issubclass(t, fault.RealType)

def is_binary_analog(x):
    t = get_type(x)
    return issubclass(t, BinaryAnalogType)

def is_bit(x):
    t = get_type(x)
    if is_binary_analog(t):
        return False
    return issubclass(t, magma.Bit)

def is_array(x):
    t = type(x) if isinstance(x, magma.Type) else x
    return issubclass(t, magma.Array)

'''
NOTE these functions are a little weird:
When magma instantiates a port as part of a circuit declaration it explicitly
flips the direction of the port. These functions take that into account. That
means they will give the wrong answer if the user manually instantiates one
of the real_types and does not flip it!
'''
def is_input(x):
    if isinstance(x, magma.Type):
        return not x.is_input()
    else:
        return x.is_input()
def is_output(x):
    if isinstance(x, magma.Type):
        return not x.is_output()
    else:
        return x.is_output()

def get_name(x):
    if type(x) == str:
        return x
    elif isinstance(x, magma.Type):
        n = x.name
        if isinstance(n, magma.ArrayRef):
            # TODO for a nested bus will the indices be the wrong order?
            bus = get_name(n.array)
            index = n.index
            return f'{bus}<{index}>'
        return n.name
    else:
        return getattr(x, 'name', None)


''' Make more friendly type names for .yaml files '''
bit = Bit
real = Real
input = magma.In
output = magma.Out
binary_analog = BinaryAnalog
