import fixture
from fixture import Regression
from fixture import signals
import fault, magma
import random

from fixture.signals import SignalManager

rand = random.random

def get_simple_amp():
    class UserAmpInterface(magma.Circuit):
        name = 'my_simple_amp_interface'
        IO = [
            'my_in', fault.RealIn,
            'my_out', fault.RealOut,
            'vdd', fault.RealIn,
            'vss', fault.RealIn,
        ]

    test = UserAmpInterface.my_in

    mapping = {
        "in_single": "my_in",
        "out_single": "my_out"
    }
    return UserAmpInterface, mapping

def get_parameterized_amp():
    assert False, 'Have not debugged this yet'
    class Parameterized(fixture.templates.SimpleAmpTemplate):
        name = 'my_parameterized_amp'
        IO = [
            'my_in', fault.RealIn,
            'my_out', fault.RealOut,
            'vdd', fault.RealIn,
            'vss', fault.RealIn,
            'ba<0>', magma.BitIn,
            'ba<1>', magma.BitIn,
            'ba<2>', magma.BitIn,
            'ba<3>', magma.BitIn,
            'adj', fault.RealIn,
            'ctrl<0>', magma.BitIn,
            'ctrl<1>', magma.BitIn,
            'vdd_internal', fault.RealOut()
        ]
        def mapping(self):
            self.in_single = self.my_in
            self.out_single = self.my_out
    return Parameterized


def test_simple_amp():
    data = ({'in_single':[1, 2, 3, 4, 5], 'amp_output':[6,4,5,2,2]})
    dut, mapping = get_simple_amp()
    s_in = signals.SignalIn(
        None,
        'real',
        None,
        None,
        'spice_in',
        None,
        'in_single',
        False
    )
    s_out = signals.SignalOut(
        'real',
        'spice_out',
        None,
        'out_single'
    )
    sm = SignalManager([s_in, s_out], {'in_single': s_in, 'out_single': s_out})
    t = fixture.templates.SimpleAmpTemplate(dut, None, sm)
    reg = Regression(t, t.tests[0], data)
    # TODO actually check the results in reg

'''
TODO update these tests to the new Style of Template
def test_parameterized_amp():
    data = ({'in_single':[1, 2, 3, 4, 5],
             'adj': [2, 5, 4, 1, 3],
             'ba<0>': [0, 0, 1, 1, 1],
             'ba<1>': [0, 1, 0, 0, 1],
             'ba<2>': [1, 0, 0, 0, 1],
             'ba<3>': [0, 0, 0, 1, 0],
             'amp_output':[6,4,5,2,2]
            })
    dut = get_parameterized_amp()
    reg = Regression(dut, data)

def test_differential_amp():
    def model_differential_amp(p, inp, inn):
        in_diff = inp - inn
        in_cm = (inp+inn)/2
        out_diff = p['gain']*in_diff + p['cm_gain']*in_cm + p['offset']
        out_cm = p['gain_to_cm']*in_diff + p['cm_gain_to_cm']*in_cm + p['offset_to_cm']
        outp = out_cm + out_diff/2
        outn = out_cm - out_diff/2
        return (outp, outn)
        
    differential_params = {
        'gain':2.5,
        'cm_gain':0.1,
        'offset':-0.01,
        'gain_to_cm':0.05,
        'cm_gain_to_cm':-0.5,
        'offset_to_cm':0.6
    }

    N = 20
    datax = {p:[] for p in ['inp', 'inn']}
    datay = {p:[] for p in ['outp', 'outn']}

    for i in range(N):
        inp = rand() * 1.2
        inn = rand() * 1.2
        outp, outn = model_differential_amp(differential_params, inp, inn)
        datax['inp'].append(inp)
        datax['inn'].append(inn)
        datay['outp'].append(outp)
        datay['outn'].append(outn)
    data = {**datax, **datay}

    print('\nDATA', data)


    class Diff(fixture.templates.DifferentialAmpTemplate):
        name = 'my_diff_thing'
        IO = ['inp', fixture.RealIn((0.0, 1.2)),
              'inn', fixture.RealIn((0.0, 1.2)),
              'outp', fixture.RealOut(),
              'outn', fixture.RealOut()
             ]
        def mapping(self):
            # we don't need to do anything because pin names already match
            pass

    reg = Regression(Diff, data)
'''



if __name__ == '__main__':
   test_simple_amp()
   #test_parameterized_amp()
   #test_differential_amp()

