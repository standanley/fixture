from fixture import LinearRegression, LinearRegressionSM
import random

def gen_data(iv_names, dv_names, N, fun, noisy=True):
    dims = len(iv_names)
    ivs = {n:[] for n in iv_names}
    dvs = {n:[] for n in dv_names}
    for i in range(N):
        xs = [random.random() for _ in range(dims)]
        for iv, x in zip(iv_names, xs):
            ivs[iv].append(x) 
        ys = list(fun(*xs))
        if noisy:
            for i in range(len(ys)):
                ys[i] += random.random()*.05
        for dv, y in zip(dv_names, ys):
            dvs[dv].append(y) 
    return ivs, dvs

def test_simple():
    fun = lambda a,b: (2*a + 3*b - 5, -1*a + 2.5*b +1.6)

    iv_names = ['a', 'b']
    dv_names = ['out1', 'out2']

    ivs, dvs = gen_data(iv_names, dv_names, 15, fun)

    test = LinearRegressionSM(dv_names, iv_names, (ivs, dvs))
    test.run()
    summary = test.get_summary()
    for dv in summary:
        print(dv, '\n', summary[dv])

    print('\n\nSuggested model:')
    suggested_formula = test.suggest_model_using_confidence_interval()
    print(suggested_formula)
    test.run(suggested_formula)
    summary = test.get_summary()
    for dv in summary:
        print(dv, '\n', summary[dv])




if __name__ == '__main__':
    test_simple()
