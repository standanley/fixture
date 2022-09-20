import itertools
import operator
import os
from functools import reduce

#import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
plt = matplotlib.pyplot
import scipy
import statsmodels.formula.api as smf
import pandas
import numpy as np
from fixture.regression import Regression
from fixture.signals import SignalIn, SignalOut, SignalArray, CenteredSignalIn
from scipy.interpolate import griddata


class PlotHelper:
    dpi = 300

    def __init__(self, test, data, parameter_algebra, regression_results, mode_prefix, expr_dataframe, expr_fit):
        self.test = test
        self.data = data
        self.parameter_algebra = parameter_algebra
        self.regression_results = regression_results
        self.mode_prefix = mode_prefix
        self.expr_dataframe = expr_dataframe
        self.expr_fit = expr_fit

    def get_column(self, target, overrides=None, lhs_pred=False, param_meas=False):
        '''
        6 possibilities:
        out = gain*in + offset
        gain and offset are functions of vdd

        1) lhs: out
        2) input: in
        3) optional in: vdd
        4) param: gain = gain_vdd_fit * vdd + gain_const_fit
        5) lhs_pred: out_lhs_pred = gain*in + offset
        6) param_meas: gain_param_meas = (out - offset) / in

        Cases 5 and 6 are accessed with the corresponding kwargs

        Also, cases [1, 4, 5, 6] can be adjusted to mimic optional input values
        Also, if target is a tuple, returns the product of the elements
        '''

        # clean up some inputs
        if overrides is None:
            overrides = {}
        #if isinstance(target, (SignalIn, SignalOut)):
        #    target = Regression.regression_name(target)

        # easy cases
        if target in overrides:
            # TODO what if override is a number, not column?
            return overrides[target]
        overrides_str = {Regression.regression_name(s): v for s, v in overrides.items()}
        if target in overrides_str:
            return overrides_str[target]
        #if isinstance(target, tuple):
        #    print('TODO remove tuple case')
        #    return reduce(operator.mul,
        #                  [self.get_column(elem, overrides) for elem in target],
        #                  1)
        if isinstance(target, SignalArray) and target.info['datatype'] == 'binary_analog':
            # TODO should have something like Sample.get_plot_value in the future
            #regression_name = Regression.regression_name(target)
            # TODO issues if regression_name(b) is in overrides?
            #if regression_name in self.data and all(b not in overrides for b in target):
            #    # no need to build the decimal version here if it's already in the data
            #    return self.data[regression_name]
            if target in self.expr_dataframe:
                return self.expr_dataframe[target]
            # I think we only get here if the user manually enters the bits but not decimal in the spreadsheet
            elements = [self.get_column(b, overrides) for b in target]
            result = target.get_decimal_value(elements)
            return result

        # now we start breaking it up by the 6 cases
        if target in self.parameter_algebra:
            if not lhs_pred:
                # Case 1) lhs
                if overrides == {}:
                    #return self.data[Regression.regression_name(target)]
                    return self.expr_dataframe[target]
                else:
                    # we want to adjust this lhs based on overrides
                    # Basically, we see how much the overrides change the rhs,
                    # and apply that change to the lhs
                    adjustment = (self.get_column(target, lhs_pred=True, overrides=overrides)
                                  - self.get_column(target, lhs_pred=True))
                    result = self.get_column(target) + adjustment
                    return result

            else:
                # Case 5) lhs_pred
                #rhs = self.parameter_algebra[target]
                #pred = reduce(operator.add,
                #              [self.get_column(param, overrides) * self.get_column(factors)
                #               for param, factors in rhs.items()])
                #return pred
                rhs_expr = self.expr_fit[target]
                val = rhs_expr.eval(self.expr_dataframe)
                return val

        #params = {p: t
        #          for rhs in self.parameter_algebra.values()
        #          for p, t in rhs.items()}
        for lhs, rhs in self.parameter_algebra.items():
            if target in rhs:
                if not param_meas:
                    # Case 4) param
                    rhs_expr = self.expr_fit[lhs]
                    target_expr = rhs_expr.search(target)
                    assert target_expr is not None, f'Could not find name "{target}" in rhs expression "{rhs_expr.name}"'
                    val = target_expr.eval(self.expr_dataframe)
                    #fit = self.regression_results[lhs][target]
                    #param = reduce(operator.add,
                    #              [self.get_column(param_comp, overrides) * coef
                    #               for param_comp, coef in fit.items()])
                    return val
                else:
                    # Case 6: gain = (out - offset) / in

                    other_terms = reduce(operator.add,
                        [self.get_column(param, overrides) * self.get_column(factors)
                         for param, factors in rhs.items() if param != target],
                        np.zeros(self.data.shape[0]))
                    this_term_factors = self.get_column(rhs[target])
                    lhs_column = self.get_column(lhs)
                    result = (lhs_column - other_terms) / this_term_factors
                    return result
        # if we are still here, it should be Case 2 or 3
        # without access to signals, we can't check if it's case 3, so we just
        # assume target will be in data and go with it

        #target = Regression.regression_name(target)
        #assert target in self.data, f"Can't find target {target}, assumed this was Case 2 or 3"
        #return self.data[target]
        assert target in self.expr_dataframe, f"Can't find target {target}, assumed this was Case 2 or 3"
        return self.expr_dataframe[target]



    @classmethod
    def clean_filename(cls, orig):
        clean = orig.replace('/', '_over_')
        return clean

    def _save_current_plot(self, name):
        self.save_current_plot(self.mode_prefix + name)

    @classmethod
    def save_current_plot(cls, name):
        plt.grid()
        #plt.show()

        os.makedirs('plots', exist_ok=True)
        plt.savefig('./plots/' + cls.clean_filename(name), dpi=cls.dpi)
        # I've had lots of problems with seeing the results of earlier plots
        # on new plots, and this might be the solution?
        # I think the issue is when a plot is created outside plot_helper,
        # then future plot_helper things are done on top of it
        plt.clf()

    #@classmethod(parameter_algebra, data, )

    def plot_regression(self):
        #models = regression.results_models
        #for reg_name, value in models.items():
        #    name = reg_name[2:-1]
        #    model = value.model
        #    inputs = model.exog
        #    measured = model.endog
        #    predicted = model.predict(value.params)
        #    #predicted = model.predict(inputs)

        #    ##fig = plt.figure()
        #    #plt.plot(measured, predicted, 'x')
        #    #start = min(0, min(measured))
        #    #end = max(0, max(measured))
        #    #plt.plot([start, end], [start, end], '--')
        #    #plt.title(name)
        #    #plt.xlabel('Measured value')
        #    #plt.ylabel('Predicted by model')
        #    #plt.grid()
        #    ##plt.show()
        #    #self._save_current_plot(f'{name}_fit')
        #    ##plt.close(fig)

        #self.get_column('amp_output_vec[0]')
        #self.get_column('input[0]')
        #self.get_column('vdd')
        ##self.get_column(('vdd', 'vdd'))
        #self.get_column('dcgain_vec[0]')
        #self.get_column('amp_output_vec[0]', lhs_pred=True)
        #self.get_column('dcgain_vec[0]', param_meas=True)

        for lhs, rhs in self.parameter_algebra.items():
            #data = model.model.exog
            #y_pred = model.model.predict(model.params)
            #y_meas = cls.eval_factor(data, lhs)
            # TODO regression doesn't support expressions on the lhs right now
            y_meas = self.get_column(lhs)
            y_pred = self.get_column(lhs, lhs_pred=True)

            inputs = {x for factors in rhs.values() for x in factors}
            if Regression.one_literal in inputs:
                inputs.remove(Regression.one_literal)

            # regression lhs vs. each input

            #optional = list(self.regression_results[lhs].values())[0].keys()
            #optional = [x for x in optional if x != Regression.one_literal]
            optional = self.test.signals.optional_expr()
            inputs_and_opt = list(inputs) + optional
            def uncenter(s):
                return s.ref if isinstance(s, CenteredSignalIn) else s
            inputs_clean = [uncenter(s) for s in inputs_and_opt]
            for input in inputs_clean:
                x = self.get_column(input)
                x_nonan = x[~np.isnan(x)]

                if len(x_nonan) != len(y_pred):
                    # something to do with a nan removing rows from y_pred
                    continue

                assert len(x) == len(y_meas)
                assert len(x_nonan) == len(y_pred)


                plt.figure()
                plt.plot(x, y_meas, '*')
                plt.plot(x_nonan, y_pred, 'x')
                plt.xlabel(self.friendly_name(input))
                plt.ylabel(lhs)
                plt.grid()
                plt.legend(['Measured', 'Predicted'])
                plt.title(f'{lhs} vs. {self.friendly_name(input)}')
                self._save_current_plot(f'{lhs}_vs_{self.friendly_name(input)}')
                #plt.show()


                # do this plot again, but correcting for influences from
                # optional inputs
                nom_dict = {}
                for opt in optional:
                    if opt == input:
                        # TODO what is this case for?
                        continue

                    # Put nominal values if nominal dict
                    # TODO nominal should be a property of the signal in the future
                    # could be in df as decimal, binary, or both (or neither, I guess)
                    #regression_name = Regression.regression_name(opt)
                    #if regression_name in self.data:
                    if opt in self.expr_dataframe:
                        # should get analog, and also digital in the dataframe
                        # as a whole
                        nominal = np.average(self.expr_dataframe[opt])
                        if isinstance(opt, SignalArray):
                            nominal = round(nominal)
                        nom_dict[opt] = nominal
                    elif isinstance(opt, SignalArray):
                        # only if the entire SignalArray was not in the
                        # dataframe do we fall back to this individual-bit thing
                        opt_flat = opt.flat
                        for opt_single in opt_flat:
                            assert opt_single in self.expr_dataframe, f'Cannot find signal {opt} in dataframe'
                            nominal = np.average(self.expr_dataframe[opt])
                            nom_dict[opt_single] = nominal
                    else:
                        assert False, f'Do not know how to deal with nominal for {opt}'


                # If there was something that can be pinned nominal, do the plot
                if len(nom_dict) != 0:
                    x_adj = self.get_column(input, overrides=nom_dict)
                    y_meas_adj = self.get_column(lhs, overrides=nom_dict)
                    y_pred_adj = self.get_column(lhs, overrides=nom_dict, lhs_pred=True)
                    plt.figure()
                    plt.scatter(x_adj, y_meas_adj, marker='*', s=20)
                    plt.scatter(x_adj, y_pred_adj, marker='x', s=20)
                    plt.xlabel(self.friendly_name(input))
                    plt.ylabel(lhs)
                    plt.grid()
                    plt.legend(['Measured', 'Predicted'])
                    plt.title(f'{lhs} vs. {self.friendly_name(input)}, corrected for nominal {[self.friendly_name(opt) for opt in optional]}')
                    #self._save_current_plot(f'{lhs}_vs_{input}_nom_{[str(opt) for opt in optional]}')
                    self._save_current_plot(f'{lhs}_vs_{self.friendly_name(input)}_nom_opt')
                    #plt.show()



            for input_pair in itertools.combinations(inputs_clean, 2):
                input1, input2 = sorted(input_pair, key=Regression.regression_name)
                x1 = self.get_column(input1)
                x2 = self.get_column(input2)
                gridx, gridy = np.mgrid[min(x1):max(x1):1000j,
                                        min(x2):max(x2):1000j]
                points = np.vstack((x1, x2)).T
                try:
                    contour_data_meas = griddata(points, y_meas, (gridx, gridy), method='linear')
                    contour_data_pred = griddata(points, y_pred, (gridx, gridy), method='linear')
                except scipy.spatial.qhull.QhullError:
                    print(f'Issue with input point collinearity for contour plot {lhs}, {input1} vs {input2}')
                    continue

                plt.figure()
                cp = plt.contourf(gridx, gridy, contour_data_meas)#, levels=5)
                plt.colorbar(cp)
                meas_levels = cp.cvalues
                meas_breaks = meas_levels[:-1] + np.diff(meas_levels)/2
                plt.contour(gridx, gridy, contour_data_meas, levels=meas_breaks, colors='black', linestyles='solid')
                plt.plot(*(points.T), 'x')
                plt.contour(gridx, gridy, contour_data_pred, levels=meas_breaks, colors='black', linestyles='dashed')
                plt.xlabel(self.friendly_name(input1))
                plt.ylabel(self.friendly_name(input2))
                plt.title(lhs)
                plt.grid()
                self._save_current_plot(
                    f'{lhs}_vs_{self.friendly_name(input1)}_and_{self.friendly_name(input2)}')
                #plt.show()


                #fig = plt.figure()
                #ax = fig.add_subplot(111, projection='3d')
                #ax.scatter(x1, x2, y_meas)
                #ax.scatter(x1, x2, y_pred)
                ## ax.scatter(in_diff, in_cm, pred_tmp)
                ##plt.show()
                #print()


    @staticmethod
    def eval_factor(data, factor):

        if factor == '1':
            return np.ones(data.shape[0])
        else:
            #clean_factor = Regression.clean_string(factor)
            term = (Regression.regression_name(x) for x in factor)
            column = reduce(operator.mul,
                            [data[x] for x in term],
                            data[Regression.one_literal])
            return column

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
        for name in orig.keys():
            #new_name = Regression.clean_string(name)
            new_name = name
            value = overrides.get(name, orig[name])
            new_dict[new_name] = value
        return new_dict #pandas.DataFrame(new_dict)

    @staticmethod
    def friendly_name(s):
        if isinstance(s, SignalIn):
            return s.friendly_name()
        return s

    def plot_optional_effects(self):

        def regression_name(s):
            return Regression.clean_string(Regression.regression_name(s))

        N = 101

        # create nominal_data_dict
        nominal_data_dict = {}
        for ta in self.test.signals.optional_true_analog():
            assert isinstance(ta.value, tuple) and len(ta.value) == 2
            nominal = sum(ta.value) / 2
            nominal_data_dict[ta] = nominal
        ba_buses = self.test.signals.optional_quantized_analog()
        ba_bits = [bit for ba_bus in ba_buses for bit in ba_bus]
        for ba in ba_bits:
            nominal_data_dict[ba] = 0.5
        nominal_data_dict[Regression.one_literal] = 1
        #nominal_data = pandas.DataFrame(nominal_data_dict)
        nominal_data = nominal_data_dict

        for opt in self.test.signals.optional_quantized_analog():
            assert opt.info['datatype'] == 'binary_analog'
            codes = np.array(list(itertools.product(range(2), repeat=len(list(opt)))))
            sweep_data = {b: xs for b, xs in zip(opt, codes.T)}
            model_data = self.modify_data(nominal_data, sweep_data)
            model_data_dec = opt.get_decimal_value(codes.T)
            #print(model_data)

            print('TODO fix plotting of optional outputs')
            #for lhs, rhs in list(self.regression_results.items())[:2]:
            #    for parameter, fit in rhs.items():
            #        if parameter == 'default_factory':
            #            # unfortunately comes up as a key in a defaultdict
            #            continue
            #        parameter_fit = self.get_column(parameter, overrides=model_data)
            #        parameter_measured = self.get_column(parameter, overrides={}, param_meas=True)
            #        opt_measured_dec = (self.data[Regression.regression_name(opt)] if Regression.regression_name(opt) in self.data
            #                            else opt.get_decimal_value(self.data[[regression_name(x) for x in opt]].T))
            #        plt.figure()
            #        plt.scatter(opt_measured_dec, parameter_measured, marker='o', s=4)
            #        plt.scatter(model_data_dec, parameter_fit, marker='+', s=4)
            #        plt.legend(['measured', 'modeled'])
            #        plt.xlabel(self.friendly_name(opt))
            #        plt.ylabel(str(parameter))
            #        plt.title(f'{parameter} vs. {self.friendly_name(opt)}, adjusted for nominal')
            #        plt.grid()
            #        self._save_current_plot(f'{parameter}_vs_{self.friendly_name(opt)}_adjfornominal')

        for opt in self.test.signals.optional_true_analog():
            assert isinstance(opt.value, tuple) and len(opt.value) == 2
            xs = np.linspace(opt.value[0], opt.value[1], N)
            model_data = self.modify_data(nominal_data,
                                                {opt: xs})

            print()

            print('TODO fix plotting of optional outputs')
            # I started to fix this, but remembered that it will probably
            # change with the new nonlinearity stuff
            return
            for s in self.test.signals.auto_measure():
                # goal is to plot s vs opt, removing other influences

                assert False
                s_data = self.get_column(s, overrides=model_data)
                opt_data = model_data[opt]



            #    parameter = s.spice_name
            #    fit = self.regression_results[parameter]
            #    # prediction just based on  parameter = (Ax + By + C)
            #    model_prediction = self.get_column(parameter, lhs_pred=True, overrides=model_data)
            #    #model_prediction = PlotHelper.eval_parameter(
            #    #    model_data, regression_results, parameter)

            #    # prediction based on measured data
            #    # OUT = (Ax + By + C) * IN + (Dx + Ey + F)
            #    # lhs_measured = (goal) * multiplicand_measured + other_terms
            #    # goal = (lhs_measured - other_terms) / multiplicand_measured
            #    M = data.shape[0]
            #    other_terms = np.zeros(M)
            #    pas = [(k, v) for k, v in test.parameter_algebra.items()
            #           if parameter in v]
            #    # TODO fix this
            #    if len(pas) == 0:
            #        print(f'Could not find pa for {parameter}')
            #        continue
            #    assert len(pas) > 0, f'Missing parameter algebra for {parameter}?'
            #    assert len(pas) == 1, f'multiple parameter algebras for {parameter}?'
            #    pa = pas[0]
            #    for p, coef in pa[1].items():
            #        if p != parameter:
            #            p_measured = PlotHelper.eval_parameter(data, regression_results, p)
            #            coef_measured = PlotHelper.eval_factor(data, coef)
            #            other_terms += p_measured * coef_measured

            #    opt_measured = PlotHelper.eval_factor(data, regression_name(opt))
            #    lhs_measured = PlotHelper.eval_factor(data, pa[0])
            #    multiplicand_measured = PlotHelper.eval_factor(data, pa[1][parameter])
            #    parameter_measured = (lhs_measured - other_terms) / multiplicand_measured

            #    # TODO remove influence of other optional pins
            #    # (Ax + Bynom + C) = (Ax + By + C) - B(y - ynom)
            #    adjustment = np.zeros(M)

            #    for s in test.signals.optional_true_analog() + ba_bits:
            #        if s != opt:
            #            y = PlotHelper.eval_factor(data, regression_name(s))
            #            # TODO I think there's a better way to get ynom
            #            ynom = PlotHelper.eval_factor(model_data, regression_name(s))[0]
            #            # We can't use spice name here because optional signals
            #            # don't necessarily correspond to pins (sampler jitter)
            #            # BUT we can't use regression_name() because it replaces
            #            # <> with __, which is bad here
            #            name_spice_preferred = (s.spice_name
            #                if s.spice_name is not None else s.template_name)
            #            B = regression_results[parameter][name_spice_preferred]
            #            adjustment += B * (y - ynom)
            #    parameter_measured_adjusted = parameter_measured - adjustment


                #plt.figure(plt.gcf().number+1)
                plt.figure()
                plt.plot(xs, model_prediction, '--')
                plt.plot(opt_measured, parameter_measured_adjusted, 'o')
                plt.xlabel(self.friendly_name(opt))
                plt.ylabel(parameter)
                self._save_current_plot(f'{parameter}_vs_{self.friendly_name(opt)}')
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
