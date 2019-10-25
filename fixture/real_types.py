import fault
import magma
from magma.port import Port, INPUT, OUTPUT, INOUT


# this is a little strange since the types are actually classes, not instances
# When the user says Thing(args) they feel like they're instantiating a class
# Really they want to be instantiating a metaclass
# But really really they're just calling a generator function

class RealKind2(fault.RealKind):
    # TODO we should probably override qualify as well
    def flip(cls):
        if cls.isoriented(INPUT):
            return RealOut(getattr(cls, 'limits', None))
        elif cls.isoriented(OUTPUT):
            return RealIn(getattr(cls, 'limits', None))
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
