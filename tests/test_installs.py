from pathlib import Path

def test_magma():
    import magma
    print(magma)

def test_fault():
    import fault
    print(fault)

def test_verilator():
    import magma as m
    import fault

    class MyBuff(m.Circuit):
        name = 'my_inv'
        IO = ['in_', m.BitIn, 'out', m.BitOut]
        @classmethod
        def definition(io):
            io.out <= io.in_

    tester = fault.Tester(MyBuff)

    tester.poke(MyBuff.in_, 0)
    tester.eval()
    tester.expect(MyBuff.out, 0)
    tester.poke(MyBuff.in_, 1)
    tester.eval()
    tester.expect(MyBuff.out, 1)

    tester.compile_and_run('verilator', flags=['-Wno-fatal'])

def test_ngspice():
    import magma as m
    import fault

    MyAmp = m.DeclareCircuit(
        'myamp',
        'in_', fault.RealIn,
        'out', fault.RealOut,
        'vdd', fault.RealIn,
        'vss', fault.RealIn
    )

    tester = fault.Tester(MyAmp)
    tester.poke(MyAmp.vss, 0)
    tester.poke(MyAmp.vdd, 1.2)
    tester.poke(MyAmp.in_, 0.7)
    tester.expect(MyAmp.out, .81, above=0.81, below=0.82)

    tester.poke(MyAmp.in_, 0.8)
    tester.expect(MyAmp.out, .5, above=0.50, below=0.51)

    tester.compile_and_run('spice',
        simulator='ngspice', 
        model_paths = [Path('tests/spice/myamp.sp').resolve()]
    )
