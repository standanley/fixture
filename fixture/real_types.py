import fault
import magma
from magma.port import Port, INPUT, OUTPUT, INOUT


# this is a little strange since the types are actually classes, not instances


def RealIn(limits=None):
    temp = fault.real_type.MakeReal(direction=magma.port.INPUT)
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
