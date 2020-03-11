import fault
import magma
#from magma.port import Port, Direction.In, Direction.Out, Direction.InOut
from magma.t import Direction


# this is a little strange since the types are actually classes, not instances
# When the user says Thing(args) they feel like they're instantiating a class
# Really they want to be instantiating a metaclass
# But really really they're just calling a generator function

class RealKind2(fault.RealKind):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

    #@classmethod
    def flip(self):
        if self.is_oriented(Direction.In):
            temp = RealOut(getattr(self, 'limits', None))
            if hasattr(self, 'name'):
                temp.name = self.name
            return temp
        elif self.isoriented(Direction.Out):
            temp = RealIn(getattr(self, 'limits', None))
            if hasattr(self, 'name'):
                temp.name = self.name
            return temp
        return self

    def qualify(cls, direction):
        if direction is None:
            return Real(getattr(cls, 'limits', None))
        elif direction == Direction.In:
            return RealIn(getattr(cls, 'limits', None))
        elif direction == Direction.Out:
            return RealOut(getattr(cls, 'limits', None))
        elif direction == Direction.InOut:
            raise NotImplementedError
            #return RealInOut(getattr(cls, 'limits', None))
        return cls

class RealType2(fault.real_type.RealType):
    def __eq__(self, rhs):
        return self.name == rhs.name
    __hash__ = magma.Type.__hash__

def RealIn(limits=None):
    kwargs = {'direction':magma.Direction.In}
    temp = RealKind2('Real', (RealType2,), kwargs)
    temp.limits = limits
    return temp

def RealOut(limits=None):
    kwargs = {'direction':magma.Direction.Out}
    temp = RealKind2('Real', (RealType2,), kwargs)
    temp.limits = limits
    return temp

def Real(limits=None):
    kwargs = {}
    temp = RealKind2('Real', (RealType2,), kwargs)
    temp.limits = limits
    return temp

'''
class BinaryAnalogKind(magma.DigitalMeta):
    @classmethod
    def qualify(mcs, direction):
        if direction is None:
            return BinaryAnalog
        elif direction == Direction.In:
            return BinaryAnalogIn
        elif direction == Direction.Out:
            return BinaryAnalogOut
        elif direction == Direction.InOut:
            return BinaryAnalogInOut
        return mcs

    @classmethod
    def flip(mcs):
        if mcs.isoriented(Direction.In):
            return BinaryAnalogOut
        elif mcs.isoriented(Direction.Out):
            return BinaryAnalogIn
        return mcs


def MakeBinaryAnalog(**kwargs):
    return BinaryAnalogKind('BinaryAnalog', (BinaryAnalogKind, magma.Bit), kwargs)


# TODO this is ugly now because BinaryAnalog is a funciton to return the type,
# while BinaryAnalogIn is just the type itself
def BinaryAnalog(limits=None):
    assert limits==None, 'Bit type cannot have limits'
    return MakeBinaryAnalog()

#BinaryAnalog = MakeBinaryAnalog()
BinaryAnalogIn = MakeBinaryAnalog(direction=Direction.In)
BinaryAnalogOut = MakeBinaryAnalog(direction=Direction.Out)
BinaryAnalogInOut = MakeBinaryAnalog(direction=Direction.InOut)

'''

class BinaryAnalogType(magma.Bit):
    #def __init__(self):
    #    super(BinaryAnalog, self).__init__()
    pass

def BinaryAnalog(limits=None):
    assert limits==None, 'Bit type cannot have limits'
    return BinaryAnalogType

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
