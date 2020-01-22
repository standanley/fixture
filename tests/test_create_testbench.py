import fixture
import fault
from pathlib import Path
import random

def transpose(x):
    return list(zip(*list(x)))

def transpose_special(vecs, xy):
    # comes in as
    # mode: [in,out]: {pin: [x1, ...]}
    # only use 0th mode
    vecs = {str(k):v for k,v in vecs[0].items()}
    xss = vecs[xy[0]]
    yss = vecs[xy[1]]
    return [xss, yss]

def plot(results, xy):
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        res = transpose_special(results, xy)
        plt.plot(res[0], res[1], '*')
        plt.show()


def simple_amp_tester(vectors):

    # this interface can be used for spice sims as well as verilog models
    class UserAmpInterface(fixture.templates.SimpleAmpTemplate):
        name = 'my_simple_amp_interface'
        extras = {'approx_settling_time':1e-3}
        IO = [
            'in_', fixture.RealIn((.2,1.0)),
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

    vectors = [{getattr(MyAmp, k):v for k,v in vs.items()} for vs in vectors]


    tester = fault.Tester(MyAmp)
    testbench = fixture.Testbench(tester)
    print('About to apply vectors')
    testbench.set_test_vectors(vectors)
    testbench.create_test_bench()
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path('tests/spice/myamp.sp').resolve()],
        clock_step_delay=0
    )
    results = testbench.get_results()
    #print(inputs_outputs)
    return results

def test_tiny():
    # one list of vectors for each digital mode
    vectors = [{'in_':[0.7, 0.8]}]
    results = simple_amp_tester(vectors)
    print(results)

def test_many():
    vectors = [{'in_':[random.random() for _ in range(20)]}]
    results = simple_amp_tester(vectors)
    #print(results[:10])
    #print(results)
    plot(results, ('in_', 'amp_output'))

def test_with_sampler():
    vectors = fixture.Sampler.get_orthogonal_samples(1, 0, 20)
    ports = ['in_']
    vectors = [{p:vs for p,vs in zip(ports, zip(*vectors))}]
    print('About to print vectors')
    print(vectors)

    #exit()
    results = simple_amp_tester(vectors)
    #print(results[:10])
    #print(results)
    plot(results, ('in_', 'amp_output'))


if __name__ == '__main__':
    print('Starting tests')
    test_with_sampler()
    #test_many()
    print('done')
print('test')

