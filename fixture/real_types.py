import fault
import magma

# this is a little strange since the types are actually classes, not instances


def RealIn(limits=None):
    temp = fault.real_type.MakeReal(direction=magma.port.INPUT)
    temp.limits = limits
    return temp
