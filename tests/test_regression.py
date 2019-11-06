import fixture
from fixture import Regression
import fault, magma

def get_simple_amp():
    class UserAmpInterface(fixture.templates.SimpleAmpTemplate):
        name = 'my_simple_amp_interface'
        IO = [
            'my_in', fixture.RealIn((.5,.7)),
            'my_out', fixture.RealOut(),
            'vdd', fixture.RealIn(1.2),
            'vss', fixture.RealIn(0.0),
        ]
        def mapping(self):
            self.in_single = self.my_in
            self.out_single = self.my_out
    return UserAmpInterface

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


def test_simple_amp():
    data = ({'amp_input':[1, 2, 3, 4, 5]}, {'amp_output':[6,4,5,2,2]})
    dut = get_simple_amp()
    reg = Regression(dut, data)

def test_parameterized_amp():
    data = ({'amp_input':[1, 2, 3, 4, 5],
             'adj': [2, 5, 4, 1, 3],
             'ba<0>': [0, 0, 1, 1, 1],
             'ba<1>': [0, 1, 0, 0, 1],
             'ba<2>': [1, 0, 0, 0, 1],
             'ba<3>': [0, 0, 0, 1, 0],
             }, 
            {'amp_output':[6,4,5,2,2]
            })
    dut = get_parameterized_amp()
    reg = Regression(dut, data)


if __name__ == '__main__':
   #test_simple_amp()
   test_parameterized_amp()

