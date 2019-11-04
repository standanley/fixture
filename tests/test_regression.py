import fixture
from fixture import Regression
import fault, magma

def get_parameterized_amp():
    class UserAmpInterface(fixture.templates.SimpleAmpTemplate):
        name = 'my_simple_amp_interface'
        IO = [
            'my_in', fixture.RealIn((.5,.7)),
            'my_out', fixture.RealOut(),
            'vdd', fixture.RealIn(1.2),
            'vss', fixture.RealIn(0.0),
            'ba', fixture.Array(4, fixture.input(fixture.BinaryAnalog())),
            'adj', fixture.RealIn((.45,.55)),
            'ctrl', fixture.input(magma.Bits[2]),
            'vdd_internal', fixture.RealOut()
        ]
        def mapping(self):
            self.in_single = self.my_in
            self.out_single = self.my_out
    return UserAmpInterface


def test1():
    data = ({'in_':[1, 2, 3, 4, 5]}, {'out':[6,4,5,2,2]})
    params_algebra = 'out ~ gain:in_ + offset'

    dut = get_parameterized_amp()
    expr = Regression.get_optional_pin_expression(dut)
    for t in expr.split('+'):
        print(t)
    exit()

    Regression(data, params_algebra)

if __name__ == '__main__':
   test1()

