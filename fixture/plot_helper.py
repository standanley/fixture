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
        # I've had lots of problems with seeing the results of earlier plots
        # on new plots, and this might be the solution?
        # I think the issue is when a plot is created outside plot_helper,
        # then future plot_helper things are done on top of it
        plt.clf()

    @classmethod
    def plot_regression(cls, regression, parameter_algebra, data):
        models = regression.results_models
        for reg_name, value in models.items():
            name = reg_name[2:-1]
            model = value.model
            inputs = model.exog
            measured = model.endog
            predicted = model.predict(value.params)
            #predicted = model.predict(inputs)

            ##fig = plt.figure()
            #plt.plot(measured, predicted, 'x')
            #start = min(0, min(measured))
            #end = max(0, max(measured))
            #plt.plot([start, end], [start, end], '--')
            #plt.title(name)
            #plt.xlabel('Measured value')
            #plt.ylabel('Predicted by model')
            #plt.grid()
            ##plt.show()
            #cls.save_current_plot(f'{name}_fit')
            ##plt.close(fig)


        for model, pa in zip(models.values(), parameter_algebra.items()):
            lhs, rhs = pa
            #data = model.model.exog
            y_pred = model.model.predict(model.params)
            y_meas = cls.eval_factor(data, lhs)
            for parameter, coefficient in rhs.items():
                x = cls.eval_factor(data, coefficient)
                #x = coefficient
                plt.figure()
                plt.plot(x, y_meas, '*')
                plt.plot(x, y_pred, 'x')
                plt.xlabel(coefficient)
                plt.ylabel(lhs)
                plt.grid()
                plt.legend(['Measured', 'Predicted'])
                #plt.show()
                print('in PloHelper')
                cls.save_current_plot(f'{lhs}_vs_{coefficient}')


    @staticmethod
    def eval_factor(data, factor):

        if factor == '1':
            return np.ones(data.shape[0])
        else:
            clean_factor = Regression.clean_string(factor)
            return data[clean_factor]

    @staticmethod
    def eval_parameter(data, regression_results, parameter):
        fit = regression_results[parameter]
        M = data.shape[0]
        result = np.zeros(M)
        for expr, coef in fit.items():
            term = PlotHelper.eval_factor(data, expr)
            result += term * coef
        return result

    @staticmethod
    def modify_data(orig, overrides):
        # copy original dataframe and replace overrides
        new_dict = {}
        for name in orig.columns:
            new_name = Regression.clean_string(name)
            value = overrides.get(name, orig[name])
            new_dict[new_name] = value
        return pandas.DataFrame(new_dict)

    @classmethod
    def plot_optional_effects(cls, test, data, regression_results):

        def regression_name(s):
            return Regression.clean_string(Regression.regression_name(s))

        N = 101
        nominal_data_dict = {}
        for ta in test.signals.optional_true_analog():
            assert isinstance(ta.value, tuple) and len(ta.value) == 2
            nominal = sum(ta.value) / 2
            nominal_data_dict[regression_name(ta)] = [nominal] * N
        ba_buses = test.signals.optional_quantized_analog()
        ba_bits = [bit for ba_bus in ba_buses for bit in ba_bus]
        for ba in ba_bits:
            nominal_data_dict[regression_name(ba)] = [0.5] * N
        nominal_data_dict[Regression.one_literal] = [1]*N
        nominal_data = pandas.DataFrame(nominal_data_dict)


        for opt in test.signals.optional_true_analog():
            assert isinstance(opt.value, tuple) and len(opt.value) == 2
            xs = np.linspace(opt.value[0], opt.value[1], N)
            model_data = PlotHelper.modify_data(nominal_data,
                                                {regression_name(opt): xs})

            for parameter, fit in regression_results.items():
                # prediction just based on  parameter = (Ax + By + C)
                model_prediction = PlotHelper.eval_parameter(
                    model_data, regression_results, parameter)

                # prediction based on measured data
                # OUT = (Ax + By + C) * IN + (Dx + Ey + F)
                # lhs_measured = (goal) * multiplicand_measured + other_terms
                # goal = (lhs_measured - other_terms) / multiplicand_measured
                M = data.shape[0]
                other_terms = np.zeros(M)
                pas = [(k, v) for k, v in test.parameter_algebra.items()
                       if parameter in v]
                assert len(pas) == 1, f'multiple parameter algebras for {parameter}?'
                pa = pas[0]
                for p, coef in pa[1].items():
                    if p != parameter:
                        p_measured = PlotHelper.eval_parameter(data, regression_results, p)
                        coef_measured = PlotHelper.eval_factor(data, coef)
                        other_terms += p_measured * coef_measured

                opt_measured = PlotHelper.eval_factor(data, regression_name(opt))
                lhs_measured = PlotHelper.eval_factor(data, pa[0])
                multiplicand_measured = PlotHelper.eval_factor(data, pa[1][parameter])
                parameter_measured = (lhs_measured - other_terms) / multiplicand_measured

                # TODO remove influence of other optional pins
                # (Ax + Bynom + C) = (Ax + By + C) - B(y - ynom)
                adjustment = np.zeros(M)

                for s in test.signals.optional_true_analog() + ba_bits:
                    if s != opt:
                        y = PlotHelper.eval_factor(data, regression_name(s))
                        # TODO I think there's a better way to get ynom
                        ynom = PlotHelper.eval_factor(model_data, regression_name(s))[0]
                        # We can't use spice name here because optional signals
                        # don't necessarily correspond to pins (sampler jitter)
                        # BUT we can't use regression_name() because it replaces
                        # <> with __, which is bad here
                        name_spice_preferred = (s.spice_name
                            if s.spice_name is not None else s.template_name)
                        B = regression_results[parameter][name_spice_preferred]
                        adjustment += B * (y - ynom)
                parameter_measured_adjusted = parameter_measured - adjustment


                plt.figure(plt.gcf().number+1)
                plt.plot(xs, model_prediction, '--')
                plt.plot(opt_measured, parameter_measured_adjusted, 'o')
                plt.xlabel(opt.spice_name)
                plt.ylabel(parameter)
                cls.save_current_plot(f'{parameter}_vs_{opt.spice_name}')
                plt.grid()
                #plt.show()

    @classmethod
    def plot_optional(cls, x_name, y_name):
        # goal is to plot y vs x
        # e.g. diff_out vs diff_in
        # BUT we need to control for optional inputs, e.g. vdd
        # out_diff = (A+B*vdd)*in_diff + (C+D*vdd)
        # out_diff_nom = out_diff + (vdd_nom-vdd)*B*in_diff + (vdd-vdd_nom)*D
        pass



if __name__ == '__main__':
    x = [1, 2, 3]
    y = [6, 5, 2]
    plt.plot(x, y)
    default_dpi = plt.gcf().dpi
    plt.savefig('myfig2', dpi=default_dpi*3)
    x = plt.gcf()
    print('testing')