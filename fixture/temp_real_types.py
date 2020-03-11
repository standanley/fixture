import fault
import magma
from magma.t import Direction

class RealType2(fault.RealType):
    def __init__(self, limits, *largs, **kwargs):
        super().__init__(*largs, **kwargs)
        self.limits = limits




def RealOut(limits=None):
    kwargs = {'direction':magma.port.Direction.Out}
    temp = RealKind2('Real', (RealType2,), kwargs)
    temp.limits = limits
    return temp

def Real(limits=None):
    kwargs = {}
    temp = RealKind2('Real', (RealType2,), kwargs)
    temp.limits = limits
    return temp


class BinaryAnalogKind(magma.BitKind):
    def qualify(cls, direction):
        if direction is None:
            return BinaryAnalog
        elif direction == Direction.IN:
            return BinaryAnalogIn
        elif direction == Direction.Out:
            return BinaryAnalogOut
        elif direction == Direction.InOut:
            return BinaryAnalogInOut
        return cls

    def flip(cls):
        if cls.isoriented(Direction.IN):
            return BinaryAnalogOut
        elif cls.isoriented(Direction.Out):
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
BinaryAnalogIn = MakeBinaryAnalog(direction=Direction.IN)
BinaryAnalogOut = MakeBinaryAnalog(direction=Direction.Out)
BinaryAnalogInOut = MakeBinaryAnalog(direction=Direction.InOut)

def Bit(limits=None):
    assert limits==None, 'Bit type cannot have limits'
    return magma.Bit

def Array(n, t):
    return magma.Array[n, t]


def TestVectorInput(limits=None, name='Unnamed test vector input', binary_analog=False):
        temp = RealIn(limits)
        temp.name = name
        temp.binary_analog = binary_analog
        return temp

def TestVectorOutput(name='Unnamed test vector output'):
    temp = RealIn()
    temp.name = name
    return temp


''' Make more acceptable type names for .yaml files '''
bit = Bit
real = Real
input = magma.In
output = magma.Out
binary_analog = BinaryAnalog
