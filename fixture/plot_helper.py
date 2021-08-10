import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas
import numpy as np
from fixture.regression import Regression


class PlotHelper:
    dpi = 600

    @classmethod
    def save_current_plot(cls, name):
        plt.grid()
        plt.savefig(name, dpi=cls.dpi)

    @classmethod
    def plot_regression(cls, regression):
        models = regression.results_models
        for reg_name, value in models.items():
            name = reg_name[2:-1]
            model = value.model
            inputs = model.exog
            measured = model.endog
            predicted = model.predict(value.params)
            #predicted = model.predict(inputs)

            plt.plot(measured, predicted, 'x')
            start = min(0, min(measured))
            end = max(0, max(measured))
            plt.plot([start, end], [start, end], '--')
            plt.title(name)
            plt.xlabel('Measured value')
            plt.ylabel('Predicted by model')
            plt.grid()
            plt.show()
            cls.save_current_plot(f'{name}_fit')

    @classmethod
    def plot_optional_effects(cls, test, results, regression_results):
        def eval_factor(data, factor):
            # TODO ols is not the right thing here since we really only need the
            # exog evaluation. Internally in statsmodels there's something like
            # _eval_factor, but I don't think its' user-facing
            factor_wrapped = '1' if factor == '1' else f'I({factor})'
            x = smf.ols(f'1 ~ {factor_wrapped}', data)
            return x.exog[:, -1]

        def eval_parameter(data, parameter):
            fit = regression_results[parameter]
            M = data.shape[0]
            result = np.zeros(M)
            for expr, coef in fit.items():
                term = eval_factor(data, expr)
                result += term * coef
            return result


        #TODO a create_fake_data method

        def regression_name(s):
            return Regression.clean_string(Regression.regression_name(s))

        #test_item = list(regression_results['dcgain'])[2]
        #test_data = {'ibias': [1, 2, 3, 4, 5]}
        #test_data_pandas = pandas.DataFrame(test_data)
        #x = eval_factor(test_data_pandas, test_item)

        # analog
        for opt in test.signals.true_analog():
            assert isinstance(opt.value, tuple) and len(opt.value) == 2
            N = 101
            xs = np.linspace(opt.value[0], opt.value[1], N)

            fake_data = {}
            for ta in test.signals.true_analog():
                if ta == opt:
                    fake_data[ta.spice_name] = xs
                else:
                    assert isinstance(ta.value, tuple) and len(ta.value) == 2
                    nominal = sum(ta.value) / 2
                    fake_data[ta.spice_name] = [nominal]*N

            ba_dict = test.signals.binary_analog()
            ba_bits = [bit for ba_bus in ba_dict.values() for bit in ba_bus]
            for ba in ba_bits:
                fake_data[ba.spice_name] = [0.5]*N

            fake_data_renamed = {Regression.clean_string(k): v
                                 for k, v in fake_data.items()}
            fake_data_pandas = pandas.DataFrame(fake_data_renamed)

            results_renamed = {regression_name(k): v
                                 for k, v in results.items()}
            results_pandas = pandas.DataFrame(results_renamed)
            for parameter, fit in regression_results.items():
                # prediction just based on (Ax + By + C)
                result = eval_parameter(fake_data_pandas, parameter)

                # prediction based on measured data
                # OUT = (Ax + By + C) * IN + (Dx + Ey + F)
                # (Ax + Bynom + C) = (OUT - (Dx + Ey + F)) / IN - (By - Bynom)
                M = len(list(results.values())[0])
                other_terms = np.zeros(M)
                pas = [(k, v) for k, v in test.parameter_algebra.items()
                       if parameter in v]
                assert len(pas) == 1
                pa = pas[0]
                for p, coef in pa[1].items():
                    if p != parameter:
                        p_measured = eval_parameter(results_pandas, p)
                        coef_measured = eval_factor(results_pandas, coef)
                        other_terms += p_measured * coef_measured

                opt_measured = eval_factor(results_pandas, regression_name(opt))
                lhs_measured = eval_factor(results_pandas, pa[0])
                multiplicand_measured = eval_factor(results_pandas, pa[1][parameter])
                parameter_measured = (lhs_measured - other_terms) / multiplicand_measured
                # TODO remove influence of other optional pins

                check_this_param = eval_parameter(results_pandas, parameter)
                check_this_term = check_this_param * multiplicand_measured
                check_all_terms = check_this_term + other_terms
                check_err = lhs_measured - check_all_terms

                plt.plot(xs, result, '--')
                plt.plot(opt_measured, parameter_measured, 'o')
                plt.xlabel(opt.spice_name)
                plt.ylabel(parameter)
                #cls.save_current_plot(f'{parameter}_vs_{opt.spice_name}')
                plt.grid()
                plt.show()



if __name__ == '__main__':
    x = [1, 2, 3]
    y = [6, 5, 2]
    plt.plot(x, y)
    default_dpi = plt.gcf().dpi
    plt.savefig('myfig2', dpi=default_dpi*3)
    x = plt.gcf()
    print('testing')