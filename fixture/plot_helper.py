import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas
import numpy as np
from fixture.regression import Regression


class PlotHelper:
    dpi = 600

    @classmethod
    def clean_filename(cls, orig):
        clean = orig.replace('/', '_over_')
        return clean

    @classmethod
    def save_current_plot(cls, name):
        plt.grid()
        plt.savefig(cls.clean_filename(name), dpi=cls.dpi)

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

            plt.figure(plt.gcf().number + 1)
            plt.plot(measured, predicted, 'x')
            start = min(0, min(measured))
            end = max(0, max(measured))
            plt.plot([start, end], [start, end], '--')
            plt.title(name)
            plt.xlabel('Measured value')
            plt.ylabel('Predicted by model')
            cls.save_current_plot(f'{name}_fit')
            #plt.show()

    @classmethod
    def plot_optional_effects(cls, test, results, regression_results):
        def eval_factor(data, factor):
            # TODO ols is not the right thing here since we really only need the
            # exog evaluation. Internally in statsmodels there's something like
            # _eval_factor, but I don't think its' user-facing
            factor_wrapped = '1' if factor == '1' else f'I({factor.replace(":", "*")})'
            # also, it doesn't like infinities, which I don't really understand
            data_finite = data.copy()
            data_finite[np.logical_not(np.isfinite(data))] = 0
            x = smf.ols(f'1 ~ {factor_wrapped}', data_finite)
            return x.exog[:, -1]

        def eval_parameter(data, parameter):
            fit = regression_results[parameter]
            M = data.shape[0]
            result = np.zeros(M)
            for expr, coef in fit.items():
                term = eval_factor(data, expr)
                result += term * coef
            return result

        def modify_data(orig, overrides):
            # copy original dataframe and replace overrides
            new_dict = {}
            for name in orig.columns:
                new_name = Regression.clean_string(name)
                value = overrides.get(name, orig[name])
                new_dict[new_name] = value
            return pandas.DataFrame(new_dict)

        def regression_name(s):
            return Regression.clean_string(Regression.regression_name(s))

        N = 101
        nominal_data_dict = {}
        for ta in test.signals.true_analog():
            assert isinstance(ta.value, tuple) and len(ta.value) == 2
            nominal = sum(ta.value) / 2
            nominal_data_dict[regression_name(ta)] = [nominal] * N
        ba_dict = test.signals.binary_analog()
        ba_bits = [bit for ba_bus in ba_dict.values() for bit in ba_bus]
        for ba in ba_bits:
            nominal_data_dict[regression_name(ba)] = [0.5] * N
        nominal_data = pandas.DataFrame(nominal_data_dict)

        results_renamed = {regression_name(k): v
                           for k, v in results.items()}
        data = pandas.DataFrame(results_renamed)

        for opt in test.signals.true_analog():
            assert isinstance(opt.value, tuple) and len(opt.value) == 2
            xs = np.linspace(opt.value[0], opt.value[1], N)
            model_data = modify_data(nominal_data, {regression_name(opt): xs})

            for parameter, fit in regression_results.items():
                # prediction just based on (Ax + By + C)
                model_prediction = eval_parameter(model_data, parameter)

                # prediction based on measured data
                # OUT = (Ax + By + C) * IN + (Dx + Ey + F)
                # lhs_measured = (goal) * multiplicand_measured + other_terms
                M = len(list(results.values())[0])
                other_terms = np.zeros(M)
                pas = [(k, v) for k, v in test.parameter_algebra.items()
                       if parameter in v]
                assert len(pas) == 1, f'multiple parameter algebras for {parameter}?'
                pa = pas[0]
                for p, coef in pa[1].items():
                    if p != parameter:
                        p_measured = eval_parameter(data, p)
                        coef_measured = eval_factor(data, coef)
                        other_terms += p_measured * coef_measured

                opt_measured = eval_factor(data, regression_name(opt))
                lhs_measured = eval_factor(data, pa[0])
                multiplicand_measured = eval_factor(data, pa[1][parameter])
                parameter_measured = (lhs_measured - other_terms) / multiplicand_measured

                # TODO remove influence of other optional pins
                # (Ax + Bynom + C) = (Ax + By + C) - B(y - ynom)
                adjustment = np.zeros(M)

                for s in test.signals.true_analog() + ba_bits:
                    if s != opt:
                        y = eval_factor(data, regression_name(s))
                        # TODO I think there's a better way to get ynom
                        ynom = eval_factor(model_data, regression_name(s))[0]
                        B = regression_results[parameter][regression_name(s)]
                        adjustment += B * (y - ynom)
                parameter_measured_adjusted = parameter_measured - adjustment

                plt.figure(plt.gcf().number + 1)
                plt.plot(xs, model_prediction, '--')
                plt.plot(opt_measured, parameter_measured_adjusted, 'o')
                plt.xlabel(opt.spice_name)
                plt.ylabel(parameter)
                cls.save_current_plot(f'{parameter}_vs_{opt.spice_name}')
                #plt.show()



if __name__ == '__main__':
    x = [1, 2, 3]
    y = [6, 5, 2]
    plt.plot(x, y)
    default_dpi = plt.gcf().dpi
    plt.savefig('myfig2', dpi=default_dpi*3)
    x = plt.gcf()
    print('testing')
