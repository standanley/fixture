import fault
import magma
from magma.port import Port, INPUT, OUTPUT, INOUT


# this is a little strange since the types are actually classes, not instances
# When the user says Thing(args) they feel like they're instantiating a class
# Really they want to be instantiating a metaclass
# But really really they're just calling a generator function

class RealKind2(fault.RealKind):
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

def RealIn(limits=None):
    kwargs = {'direction':magma.port.INPUT}
    temp = RealKind2('Real', (fault.real_type.RealType,), kwargs)
    temp.limits = limits
    return temp

def RealOut(limits=None):
    kwargs = {'direction':magma.port.OUTPUT}
    temp = RealKind2('Real', (fault.real_type.RealType,), kwargs)
    temp.limits = limits
    return temp

def Real(limits=None):
    print('creating a real with no direction and limits', limits)
    kwargs = {}
    temp = RealKind2('Real', (fault.real_type.RealType,), kwargs)
    temp.limits = limits
    return temp



'''
class LinearBit(magma.Bit):
    # this doesn't work because init is never called
    #def __init__(self, *largs, **kwargs):
    #    super().__init(largs, kwargs)
    #    self._is_linear_bit = True
    pass
'''

class LinearBitKind(magma.BitKind):
    def qualify(cls, direction):
        if direction is None:
            return LinearBit
        elif direction == INPUT:
            return LinearBitIn
        elif direction == OUTPUT:
            return LinearBitOut
        elif direction == INOUT:
            return LinearBitInOut
        return cls

    def flip(cls):
        if cls.isoriented(INPUT):
            return LinearBitOut
        elif cls.isoriented(OUTPUT):
            return LinearBitIn
        return cls


def MakeLinearBit(**kwargs):
    return LinearBitKind('LinearBit', (magma.BitType,), kwargs)


LinearBit = MakeLinearBit()
LinearBitIn = MakeLinearBit(direction=INPUT)
LinearBitOut = MakeLinearBit(direction=OUTPUT)
LinearBitInOut = MakeLinearBit(direction=INOUT)

def Bit(limits=None):
    assert limits==None, 'Bit type cannot have limits'
    return magme.Bit

''' Make more acceptable type names for .yaml files '''
bit = Bit
real = Real
input = magma.In
output = magma.Out
