import fixture
import fault
import magma
from pathlib import Path
import pytest
import os

def file_relative_to_test(fname):
    return os.path.join(os.path.dirname(__file__), fname)

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

@pytest.mark.skip('Has not been updated to new TemplateMaster instatiation style')
def test_simple():
    print('\nTop of test')

    # TODO update to new style of magma IO declaration
    class MyAmp(magma.Circuit):
        name = 'myamp'
        extras = {'approx_settling_time':1e-3}
        IO = [
            'in_', fixture.RealIn((0.4, 1.0)),
            'out', fault.RealOut,
            'vdd', fixture.RealIn(1.2),
            'vss', fixture.RealIn(0.0)
        ]
    mapping = {
        'in_single': 'in_',
        'out_single': 'out'
    }
    extras = {
        'approx_settling_time': 1e-6
    }

    def run_callback(tester):
        print('Running sim')
        tester.compile_and_run('spice',
            simulator='ngspice',
            model_paths = [Path('tests/spice/myamp.sp').resolve()],
            clock_step_delay=0
        )

    t = fixture.templates.SimpleAmpTemplate(MyAmp, mapping, run_callback, extras)
    t.go()

    print('Creating test bench')
    # auto-create vectors for 1 analog dimension
    #vectors = fixture.Sampler.get_samples_for_circuit(MyAmp, 50)

    #tester = fault.Tester(MyAmp)
    #testbench = fixture.Testbench(tester)
    #testbench.set_test_vectors(vectors)
    #testbench.create_test_bench()

    #print('Analyzing results')
    #results = testbench.get_results()
    #results_reformatted = results[0]

    #mode = 0
    #results_reformatted = results[mode]

    #regression = fixture.Regression(MyAmp, results_reformatted)

@pytest.mark.skip('Has not been updated to new TemplateMaster instatiation style')
def test_simple_parameterized():
    class UserAmp(magma.Circuit):
        name = 'myamp_params'
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

    mapping = {
        'in_single': 'my_in',
        'out_single': 'my_out'
    }
    extras = {
        'approx_settling_time': 1e-6
    }

    def run_callback(tester):
        print('Running sim')
        tester.compile_and_run('spice',
           simulator='ngspice',
           model_paths = [Path('tests/spice/myamp_params.sp').resolve()],
           clock_step_delay=0,
           tmp_dir=False
       )

    t = fixture.templates.SimpleAmpTemplate(UserAmp, mapping, run_callback, extras)
    params_by_mode = t.go()
    for mode, results in params_by_mode.items():
        print('For mode', mode)
        print('param\tterm\tcoef')
        for param, d in results.items():
            for partial_term_optional, coef in d.items():
                print('%s\t%s\t%.3e' % (param, partial_term_optional, coef))

    # print('Creating test bench')
    # # auto-create vectors for 1 analog dimension

    # tester = fault.Tester(MyAmp)
    # testbench = fixture.Testbench(tester)
    # testbench.set_test_vectors(vectors)
    # testbench.create_test_bench()

    # print(f'Running sim, {len(vectors)} test vectors')
    # tester.compile_and_run('spice',
    #     simulator='ngspice',
    #     model_paths = [Path('tests/spice/myamp_params.sp').resolve()],
    #     clock_step_delay=0
    # )

    # print('Analyzing results')
    # results = testbench.get_results()
    # mode = 0
    # results_reformatted = results[mode]

    # regression = fixture.Regression(MyAmp, results_reformatted)

def test_simple_config():
    circuit_fname = file_relative_to_test('configs/simple_amp.yaml')
    fixture.run(circuit_fname)

def test_parameterized_config():
    circuit_fname = file_relative_to_test('configs/parameterized_amp.yaml')
    fixture.run(circuit_fname)

def test_adj_config():
    circuit_fname = file_relative_to_test('configs/simple_amp_adj.yaml')
    fixture.run(circuit_fname)

@pytest.mark.skipif(not os.path.exists(file_relative_to_test('../sky130/skywater-pdk')),
                    reason='Sky130 not installed')
def test_skywater():
    circuit_fname = file_relative_to_test('configs/simple_amp_sky130.yaml')
    fixture.run(circuit_fname)

    
if __name__ == '__main__':
    #test_simple()
    #test_simple_parameterized()
    #test_skywater()
    #test_simple_config()
    test_parameterized_config()
    #test_adj_config()

