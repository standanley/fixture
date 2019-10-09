import fixture
import fault
from pathlib import Path
import random

def transpose_special(vecs):
    xss, yss = zip(*vecs)
    yss_temp = [[float(y) for y in ys] for ys in yss]
    yss = list(zip(*yss_temp))
    xss = list(zip(*xss))
    return [xss, yss]

def plot(results):
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        res = transpose_special(results)
        plt.plot(res[0][0], res[1][0], '*')
        plt.show()


def simple_amp_tester(vectors):

    # this interface can be used for spice sims as well as verilog models
    class UserAmpInterface(fixture.templates.SimpleAmpTemplate):
        name = 'my_simple_amp_interface'
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
    


    tester = fault.Tester(MyAmp)
    inputs_outputs, callback = fixture.add_vectors(tester, vectors)
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path('tests/spice/myamp.sp').resolve()]
    )
    results = callback(tester)
    #print(inputs_outputs)
    return results

def test_tiny():

    vectors = [(0.7,), (0.8,)]
    results = simple_amp_tester(vectors)
    print(results)

def test_many():
    vectors = [(random.random(),) for _ in range(20)]
    results = simple_amp_tester(vectors)
    #print(results[:10])
    #print(results)
    plot(results)

def test_with_sampler():
    vectors = fixture.Sampler.get_orthogonal_samples(1, 0, 20)
    print(vectors)
    #exit()
    results = simple_amp_tester(vectors)
    #print(results[:10])
    #print(results)
    plot(results)


if __name__ == '__main__':
    test_with_sampler()
    #test_many()
