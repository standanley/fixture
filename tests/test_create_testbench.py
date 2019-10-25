import fixture
import fault
from pathlib import Path
import random

def transpose(x):
    return list(zip(*list(x)))

def transpose_special(vecs):
    print('TRANSPOSE')
    print(len(vecs))
    print(len(vecs[0]))
    print(len(vecs[0][0]))
    print(len(vecs[0][0][0]))
    # comes in as  (TODO: the comment in create_testbench is wrong!)
    # [in,out]: mode: vec: pin: x
    ins, outs = vecs
    xss = transpose(ins[0])
    yss = transpose(outs[0])
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
    testbench = fixture.Testbench(tester)
    testbench.set_test_vectors(vectors)
    testbench.create_test_bench()
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path('tests/spice/myamp.sp').resolve()]
    )
    results = testbench.get_results()
    #print(inputs_outputs)
    return results

def test_tiny():
    # one list of vectors for each digital mode
    vectors = [[(0.7,), (0.8,)]]
    results = simple_amp_tester(vectors)
    print(results)

def test_many():
    vectors = [[(random.random(),) for _ in range(20)]]
    results = simple_amp_tester(vectors)
    #print(results[:10])
    #print(results)
    plot(results)

def test_with_sampler():
    vectors = [fixture.Sampler.get_orthogonal_samples(1, 0, 20)]
    print(vectors)
    #exit()
    results = simple_amp_tester(vectors)
    #print(results[:10])
    #print(results)
    plot(results)


if __name__ == '__main__':
    test_with_sampler()
    #test_many()

