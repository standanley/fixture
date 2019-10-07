import fixture
import fault
from pathlib import Path

def reformat(results):
    ivs = []
    dvs = []
    for result in results:
        iv, dv = result
        ivs.append(iv)
        dvs.append([float(dv_component) for dv_component in dv])
    return ivs, dvs

def get_tf(stats):
    coefs = list(stats['coef_gain']['in_'])
    def tf(x):
        y = 0
        for order, coef in enumerate(coefs):
            y += coef * x**order
        return y
    return tf

def plot(results, tf):
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
    vectors = fixture.Sampler.get_orthogonal_samples(1, 0, 20)

    tester = fault.Tester(MyAmp)
    inputs_outputs, analysis_callback = fixture.add_vectors(tester, vectors)

    print(f'Running sim, {len(vectors)} test vectors')
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path('tests/spice/myamp.sp').resolve()]
    )

    print('Analyzing results')
    results = analysis_callback(tester)
    #print(results)
    results_reformatted = reformat(results)
    #print(results_reformatted)

    iv_names = ['in_']
    dv_names = ['out']
    formula = {'out':'in_ + I(in_**2) + I(in_**3)'}
    regression = fixture.LinearRegressionSM(iv_names, dv_names, results_reformatted)
    regression.run()

    stats = regression.get_statistics()
    print(regression.get_summary()['in_'])
    tf = get_tf(stats)

    print('Plotting results')
    plot(results, tf)

    


    
if __name__ == '__main__':
    test_simple()

