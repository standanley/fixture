from fixture import LinearRegression, LinearRegressionSM
import random

def gen_data(ivs, dvs, N, fun, noisy=True):
    dims = len(ivs)
    ivs = []
    dvs = []
    for i in range(N):
        x = [random.random() for _ in range(dims)]
        ivs.append(x)
        dvs.append(list(fun(*x)))
        if noisy:
            for i in range(len(dvs[-1])):
                dvs[-1][i] += random.random()*.05
    return ivs, dvs

def test_simple():
    fun = lambda a,b: (2*a + 3*b - 5, -1*a + 2.5*b +1.6)

    iv_names = ['a', 'b']
    dv_names = ['out1', 'out2']

    ivs, dvs = gen_data(iv_names, dv_names, 15, fun)

    #temp = LinearRegression(None, None, None)

    #dv_iv = {'out':['a', 'b']}
    #f = temp._make_formula(dv_iv, None, 3, True, {})
    #print('\n\n')
    #print(f)
    #print(ivs)
    #print(dvs)

    test = LinearRegressionSM(dv_names, iv_names, (ivs, dvs))
    test.run()
    summary = test.get_summary()
    for dv in summary:
        print(dv, '\n', summary[dv])

    print('\n\nsuggested model:')
    suggested_formula = test.suggest_model_using_confidence_interval()
    print(suggested_formula)
    test.run(suggested_formula)
    summary = test.get_summary()
    for dv in summary:
        print(dv, '\n', summary[dv])




if __name__ == '__main__':
    test_simple()
