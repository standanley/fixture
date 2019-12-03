import fault
import magma
from magma.port import Port, INPUT, OUTPUT, INOUT


# this is a little strange since the types are actually classes, not instances
# When the user says Thing(args) they feel like they're instantiating a class
# Really they want to be instantiating a metaclass
# But really really they're just calling a generator function

class RealKind2(fault.RealKind):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

    def flip(cls):
        if cls.isoriented(INPUT):
            return RealOut(getattr(cls, 'limits', None))
        elif cls.isoriented(OUTPUT):
            return RealIn(getattr(cls, 'limits', None))
        return cls

    def qualify(cls, direction):
        if direction is None:
            return Real(getattr(cls, 'limits', None))
        elif direction == INPUT:
            return RealIn(getattr(cls, 'limits', None))
        elif direction == OUTPUT:
            return RealOut(getattr(cls, 'limits', None))
        elif direction == INOUT:
            raise NotImplementedError
            #return RealInOut(getattr(cls, 'limits', None))
        return cls

class RealType2(fault.real_type.RealType):
    def __eq__(self, rhs):
        return self.name == rhs.name
    __hash__ = magma.Type.__hash__

def RealIn(limits=None):
    kwargs = {'direction':magma.port.INPUT}
    temp = RealKind2('Real', (RealType2,), kwargs)
    temp.limits = limits
    return temp

def RealOut(limits=None):
    kwargs = {'direction':magma.port.OUTPUT}
    temp = RealKind2('Real', (RealType2,), kwargs)
    temp.limits = limits
    return temp

def Real(limits=None):
    kwargs = {}
    temp = RealKind2('Real', (RealType2,), kwargs)
    temp.limits = limits
    return temp



'''
class BinaryAnalog(magma.Bit):
    # this doesn't work because init is never called
    #def __init__(self, *largs, **kwargs):
    #    super().__init(largs, kwargs)
    #    self._is_linear_bit = True
    pass
'''

class BinaryAnalogKind(magma.BitKind):
    def qualify(cls, direction):
        if direction is None:
            return BinaryAnalog
        elif direction == INPUT:
            return BinaryAnalogIn
        elif direction == OUTPUT:
            return BinaryAnalogOut
        elif direction == INOUT:
            return BinaryAnalogInOut
        return cls

    def flip(cls):
        if cls.isoriented(INPUT):
            return BinaryAnalogOut
        elif cls.isoriented(OUTPUT):
            return BinaryAnalogIn
        return cls


def MakeBinaryAnalog(**kwargs):
    return BinaryAnalogKind('BinaryAnalog', (magma.BitType,), kwargs)


# TODO this is ugly now because BinaryAnalog is a funciton to return the type,
# while BinaryAnalogIn is just the type itself
def BinaryAnalog(limits=None):
    assert limits==None, 'Bit type cannot have limits'
    return MakeBinaryAnalog()

#BinaryAnalog = MakeBinaryAnalog()
BinaryAnalogIn = MakeBinaryAnalog(direction=INPUT)
BinaryAnalogOut = MakeBinaryAnalog(direction=OUTPUT)
BinaryAnalogInOut = MakeBinaryAnalog(direction=INOUT)

def Bit(limits=None):
    assert limits==None, 'Bit type cannot have limits'
    return magma.Bit

def Array(n, t):
    return magma.Array[n, t]

class TestVectorInput():
    def __init__(self, limits=None, name='Unnamed test vector input', binary_analog=False):
        assert limits != None, 'Test vector input must have limits'
        self.limits = limits
        self.name = name
        self.binary_analog = binary_analog
    def __str__(self):
        return self.name

class TestVectorOutput():
    def __init__(self, name='Unnamed test vector output'):
        self.name = name
    def __str__(self):
        return self.name

''' Make more acceptable type names for .yaml files '''
bit = Bit
real = Real
input = magma.In
output = magma.Out
binary_analog = BinaryAnalog
