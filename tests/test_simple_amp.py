import fixture
import fault
import magma
from pathlib import Path

def transpose(x):
    return list(zip(*list(x)))

def get_tf(stats, ivs):
    def tf(x):
        y = 0
        for iv in ivs:
            coefs = list(stats['coef_gain'][iv])
            for order, coef in enumerate(coefs):
                y += coef * x**order
            return y
    return tf

def plot_errors(x, y1, y2):
    import matplotlib.pyplot as plt
    for a,b,c in zip(x, y1, y2):
        d = '-g' if c > b else '-r'
        plt.plot([a, a], [b, c], d)
    plt.grid()
    plt.show()

def plot2(results, statsmodels, in_dim=0):
    if __name__ != '__main__':
        return
    xs, ys = results
    x_key = list(xs)[in_dim]
    y_key = list(ys)[0]
    xs = xs[x_key]
    ys = ys[y_key]
    estimated = statsmodels.fittedvalues
    plot_errors(xs, ys, estimated)

    

def plot(results, tf):
    if __name__ != '__main__':
        return
    import matplotlib.pyplot as plt
    xs, ys = results
    xs = transpose([x[k] for k in x])
    ys = transpose([y[k] for k in y])
    plt.plot(xs, ys, '*')
    xs.sort()
    plt.plot(xs, [tf(x[0]) for x in xs], '--')
    plt.grid()
    plt.show()

def test_simple():
    print('\nTop of test')

    # this interface can be used for spice sims as well as verilog models
    class UserAmpInterface(fixture.templates.SimpleAmpTemplate):
        name = 'my_simple_amp_interface'
        extras = {'approx_settling_time':1e-3}
        IO = [
            'in_', fixture.RealIn((0.4, 1.0)),
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
    vectors = fixture.Sampler.get_samples_for_circuit(MyAmp, 50)

    tester = fault.Tester(MyAmp)
    testbench = fixture.Testbench(tester)
    testbench.set_test_vectors(vectors)
    testbench.create_test_bench()

    print(f'Running sim, {len(vectors[0])} test vectors')
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path('tests/spice/myamp.sp').resolve()],
        clock_step_delay=0
    )

    print('Analyzing results')
    results = testbench.get_results()
    results_reformatted = results[0]

    iv_names = ['amp_input']
    dv_names = ['amp_output']
    formula = {'amp_output':'amp_input + I(amp_input**2) + I(amp_input**3)'}
    regression = fixture.LinearRegressionSM(iv_names, dv_names, results_reformatted)
    regression.run()

    stats = regression.get_statistics()
    print(regression.get_summary()['amp_output'])

    print('Plotting results')
    tf = get_tf(stats, dv_names)
    plot(results_reformatted, tf)
    #temp = regression.model_ols
    #temp = temp['out']
    #plot2(results_reformatted, temp, in_dim=5)

    
def test_simple_parameterized():
    class UserAmpInterface(fixture.templates.SimpleAmpTemplate):
        extras = {'approx_settling_time':1e-3}
        name = 'my_simple_amp_interface'
        IO = [
            'my_in', fixture.RealIn((.5,.7)),
            'my_out', fault.RealOut,
            'vdd', fixture.RealIn(1.2),
            'vss', fixture.RealIn(0.0),
            'ba', magma.Array[4, magma.In(fixture.BinaryAnalog())],
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

    print('Creating test bench')
    # auto-create vectors for 1 analog dimension
    vectors =  fixture.Sampler.get_samples_for_circuit(MyAmp, 500)

    tester = fault.Tester(MyAmp)
    testbench = fixture.Testbench(tester)
    testbench.set_test_vectors(vectors)
    testbench.create_test_bench()
    inputs_outputs = testbench.get_input_output_names()


    print(f'Running sim, {len(vectors)} test vectors')
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path('tests/spice/myamp_params.sp').resolve()],
        clock_step_delay=0
    )

    print('Analyzing results')
    results = testbench.get_results()
    mode = 0
    results_reformatted = results[mode]

    iv_names, dv_names = inputs_outputs
    regression = fixture.LinearRegressionSM(iv_names, dv_names, results_reformatted)
    regression.run()
    suggested_formula = regression.suggest_model_using_sensitivity()
    regression.run(suggested_formula)

    stats = regression.get_statistics()

    print(regression.get_summary()['amp_output'])
    #print(regression.get_summary()['vdd_internal'])

    print('Plotting results')
    temp = regression.model_ols
    temp = temp['amp_output']
    plot2(results_reformatted, temp, in_dim=0)

    
if __name__ == '__main__':
    #test_simple()
    test_simple_parameterized()

