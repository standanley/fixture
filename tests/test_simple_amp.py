import fixture
from fixture.real_types import LinearBitKind
import fault
import magma
from pathlib import Path

def reformat(results):
    ivs = []
    dvs = []
    for result in results:
        iv, dv = result
        ivs.append(iv)
        #dvs.append([float(dv_component) for dv_component in dv])
        dvs.append(dv)
    return ivs, dvs

def reformat2(results):
    # was for [in,out]: for mode: for vec: for pin: x
    # we swap the first two axes
    return list(zip(*list(results)))

def get_tf(stats, ivs):
    def tf(x):
        y = 0
        for iv in ivs:
            coefs = list(stats['coef_gain'][iv])
            for order, coef in enumerate(coefs):
                y += coef * x**order
            return y
    return tf

def plot(results, tf):
    if __name__ != '__main__':
        return
    import matplotlib.pyplot as plt
    xs, ys = zip(*results)
    xs = [x[0] for x in xs]
    ys = [y[0] for y in ys]
    plt.plot(xs, ys, '*')
    xs.sort()
    plt.plot(xs, [tf(x) for x in xs], '--')
    plt.grid()
    plt.show()

def test_simple():
    print('\nTop of test')

    # this interface can be used for spice sims as well as verilog models
    class UserAmpInterface(fixture.templates.SimpleAmpTemplate):
        name = 'my_simple_amp_interface'
        IO = [
            'in_', fixture.RealIn((.5,1.0)),
            'out', fault.RealOut,
            'vdd', fixture.RealIn(1.2),
            'vss', fixture.RealIn(0.0)
        ]
        def mapping(self):
            self.in_single = self.in_
            self.out_single = self.out

    # The name and IO here match the spice model in spice/myamp.sp
    # Since we include that file in compile_and_run, they get linked
    class MyAmp(UserAmpInterface):
        name = 'myamp'

    print('Creating test bench')
    # auto-create vectors for 1 analog dimension
    vectors = fixture.Sampler.get_samples_for_circuit(MyAmp, 5)

    tester = fault.Tester(MyAmp)
    testbench = fixture.Testbench(tester)
    testbench.set_test_vectors(vectors)
    testbench.create_test_bench()

    print(f'Running sim, {len(vectors[0])} test vectors')
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path('tests/spice/myamp.sp').resolve()]
    )

    print('Analyzing results')
    results = testbench.get_results()
    print(results)
    results_reformatted = results#reformat2(results)[0]
    print(results_reformatted)

    iv_names = ['in_']
    dv_names = ['out']
    formula = {'out':'in_ + I(in_**2) + I(in_**3)'}
    regression = fixture.LinearRegressionSM(iv_names, dv_names, results_reformatted)
    regression.run()

    stats = regression.get_statistics()
    print(regression.get_summary()['out'])
    tf = get_tf(stats, dv_names)
    tf(5)

    print('Plotting results')
    plot(results, tf)

    
def test_simple_parameterized():
    class UserAmpInterface(fixture.templates.SimpleAmpTemplate):
        name = 'my_simple_amp_interface'
        IO = [
            'my_in', fixture.RealIn((.5,.7)),
            'my_out', fault.RealOut,
            'vdd', fixture.RealIn(1.2),
            'vss', fixture.RealIn(0.0),
            'ba', magma.Array[4, magma.In(fixture.LinearBit)],
            'adj', fixture.RealIn((.45,.55)),
            'ctrl', magma.In(magma.Bits[2]),
            'vdd_internal', fault.RealOut
        ]
        def mapping(self):
            self.in_single = self.my_in
            self.out_single = self.my_out



    # The name and IO here match the spice model in spice/myamp.sp
    # Since we include that file in compile_and_run, they get linked
    class MyAmp(UserAmpInterface):
        name = 'myamp_params'

    io = MyAmp.IO

    '''
    from magma import ArrayKind, Array, Flip
    from fault import RealKind
    print(io.items())
    for name, port in io.items():
        print('\n'+name)
        #print('name type', type(name))
        #print('port type', type(port))
        #print('port:', port)
        thing = getattr(MyAmp, name)
        print('thing type', type(thing))
        print('thing:', thing)
        #print()
        #print(isinstance(port, RealKind))
        #print(isinstance(type(thing), RealKind))
        #print(hasattr(port, 'limits'))
        #print(hasattr(thing, 'limits'))
        #print(isinstance(port, ArrayKind))
        #print(isinstance(thing, Array))
        #print(Flip(thing).isinput())
        #print(thing.isinput())
        #print(Flip(port).isinput())
        print(getattr(thing, 'limits', 'no limits'))
    '''


    #temp = MyAmp.ba
    #print(temp.name)
    #print(type(temp.name))
    #print(type(temp))
    ###print(type(temp.port))
    #print(temp[1].name)
    #print(type(temp[1].name))
    #print(type(temp[1]))
    #print()
    #print(temp[1].name.array.name)
    #print(temp[1].name.index)
    ##print(type(temp[1].port))
    #exit()

    print('Creating test bench')
    # auto-create vectors for 1 analog dimension
    vectors =  fixture.Sampler.get_samples_for_circuit(MyAmp, 80)

    #print(f'length of vectors {len(vectors)}\nvectors[0][0] = {vectors[0][0]}')

    tester = fault.Tester(MyAmp)
    inputs_outputs, analysis_callback = fixture.add_vectors(tester, vectors)


    print(f'Running sim, {len(vectors)} test vectors')
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path('tests/spice/myamp_params.sp').resolve()]
    )

    print('Analyzing results')
    results = analysis_callback(tester)
    #print(inputs_outputs)
    #print(results[0])
    results_reformatted = reformat(results)

    #print('printing results')
    #for r in results:
    #    print(r)
    #print('\nA')
    #print(results_reformatted)

    iv_names, dv_names = inputs_outputs
    #formula = {'out':'in_ + I(in_**2) + I(in_**3)'}
    print('\nB')
    regression = fixture.LinearRegressionSM(iv_names, dv_names, results_reformatted)
    print('\nC')
    regression.run()
    print('\nD')

    stats = regression.get_statistics()
    for dv in ['my_out', 'vdd_internal']:
        print(f'Stats for {dv}')
        print(regression.get_summary()[dv])
    #tf = get_tf(stats)

    #print('Plotting results')
    #plot(results, tf)

    
if __name__ == '__main__':
    test_simple()
    #test_simple_parameterized()

