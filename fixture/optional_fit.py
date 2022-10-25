from abc import ABC, abstractmethod
from collections import defaultdict

from scipy.optimize import minimize
from scipy.stats import linregress
import numpy as np
import pandas

import fixture
from fixture.sampler import SampleManager
from fixture.signals import SignalArray, SignalIn
from fixture.plot_helper import plt, PlotHelper

PLOT = True

class Expression:
    x_opt = None
    last_coef_is_offset = False

    def __init__(self, name):
        # TODO when I override this, instead of calling super(), I've just been
        # rewriting this one line to set the name. Is that bad practice? probably
        self.name = name

    @property
    @abstractmethod
    def NUM_COEFFICIENTS(self):
        # a constant property that is the number of coefs to fit
        # Remember that the constant offset is separate from this count
        return self._NUM_COEFFICIENTS
    @NUM_COEFFICIENTS.setter
    def NUM_COEFFICIENTS(self, value):
        # TODO see note on input_signals
        self._NUM_COEFFICIENTS = value


    @property
    @abstractmethod
    def input_signals(self):
        # a list of signals whose values this expression depends on. Those
        # values will be passed, in the same order, to predict() and fit()
        return self._input_signals
    @input_signals.setter
    def input_signals(self, value):
        # TODO this is to allow the value to be set in __init__ ...
        # There may be a better way to implement these abstract properties
        self._input_signals = value

    #@property
    #def x_init(self):
    #    # A starting point for nonlinear fitting
    #    # I think if you have a custom fit() method this may never be used
    #    return np.ones(self.NUM_COEFFICIENTS)
    #@x_init.setter
    #def x_init(self, value):
    #    # TODO I think from now these custom getters and setters are replaced
    #    #  (for this instance) since we overwrote x_init, but I guess that's fine
    #    self.x_init = value
    x_init = None

    @abstractmethod
    def predict(self, opt_values, coefs):
        # given the values of the optional inputs and the fit coefficients,
        # calculate the linear influence of the optional inputs
        pass

    def predict_from_dict(self, opt_dict, coefs):
        # Similar to predict, but takes the optional inputs as a dictionary or
        # dataframe. This one is more convenient, but the minimizer can get
        # better performance by using predict directly
        opt_values = [opt_dict[s] for s in self.input_signals]
        opt_values = np.array(opt_values)
        ans = []
        for value_vector in opt_values.T:
            ans.append(self.predict(value_vector, coefs))
        return ans

    def fit(self, optional_data, result_data):
        # return a best-fit of the coefficients, i.e. minimize
        # predict(optional_data, coefficients) - result_data
        # return a tuple of (coefficients, offset)
        my_data = optional_data[self.input_signals]
        assert my_data.shape == (len(result_data), len(self.input_signals))
        def error(coefs):
            predict_vec = np.vectorize(self.predict, signature='(n),(m)->()')
            predictions = predict_vec(my_data, coefs)
            assert len(result_data) == len(predictions)
            errors = result_data - predictions
            e = sum(errors**2)
            return (e/len(result_data))**.5

        #test_optimizers(error, self.NUM_COEFFICIENTS)

        #def get_x_init(exp):
        #    if isinstance(exp, HeirarchicalExpression):
        #        return np.concatenate([get_x_init(e)
        #                         for e in exp.child_expressions])
        #    else:
        #        if exp.x_init is None:
        #            return np.ones(exp.NUM_COEFFICIENTS)
        #        else:
        #            assert len(exp.x_init) == exp.NUM_COEFFICIENTS
        #            return exp.x_init
        #x0 = get_x_init(self)
        x0 = self.x_init
        if x0 is None:
            x0 = np.ones(self.NUM_COEFFICIENTS)

        ## ------ TEMP --------
        #x0[6] = 1
        #x0[14] = 1
        #x0[22] = 1
        #tol = 1e-5

        #import cProfile
        #def to_profile():
        #    return minimize(error, x0)
        #cProfile.runctx('to_profile()', None, locals())
        import time
        start = time.time()
        print('Minimizing', self.name)
        print('TODO stop skipping fit')
        if False and self.name == 'out_diff':
            #x_opt = np.array([-1.23059139e+03, -6.83196691e+02, -2.24022907e+02, -1.90805353e+02,
            #   -5.96662666e+01, -4.48716108e+01,  2.83153509e+03, -7.72680027e+01,
            #   -1.08935382e+02,  2.02334246e+02, -2.11149160e+02,  1.86133922e+02,
            #   -1.19705815e+02,  1.01592209e+02,  1.79559599e-04,  9.59793043e-05,
            #   -4.33567171e-05,  9.57645151e-05, -9.48833088e-05, -9.26496371e-05,
            #   -1.40574902e-04])
            # essentially null out cm and const terms
            #x0[14] = 1e6
            #x0[22] = 1e6

            #for i in range(6):
            #    resistance = 2**i * 1e3
            #    x0[i] = 1/resistance
            ## last one should be zero, but I'm afraid of divide by zero issues
            #x0[6] = 1e-7


            ##x_opt = x0
            #result = minimize(error, x0, method='Powell', options={
            #    #'gtol': tol
            #    'ftol': 0,
            #    #'fatol': 0,
            #    'maxfev': 10**5,
            #    'disp': True
            #})
            #x_opt = result.x

            x_opt = [3.42726842e+00,  5.80524094e+10, -6.14534676e-01,  1.86709610e+01,
             5.80263269e+10, -1.92858523e+01, -6.17462353e-01,  1.23284944e+00,
             1.16065944e+03, -3.15479966e+02,  5.83937701e+10,  4.83059492e+10,
             5.80241228e+10,  5.80241513e+10,  5.80241280e+10,  1.73775268e+11,
             5.80241276e+10,  9.15012886e+02, -8.88170382e-06,  4.28260112e+08,
             -5.73351504e+20,  8.45809400e+10,  9.77142526e+09, -8.92088758e+10,
             -3.67756854e+09, -1.54204027e+03,  2.79275488e-05]


        elif False and self.name == 'out_cm':
            #x_opt = np.array([-2.19233927e+01,  2.61691713e+02,  1.38773168e+01,  1.57116746e+02,
            #    4.38110640e+02,  3.03643386e+01,  1.84852178e+02,  6.07208190e+02,
            #    6.80619246e+02,  7.28624576e+02,  6.60212312e+02,  6.29206962e+02,
            #    7.36672174e+02,  1.23052961e+03, -1.14858118e-03, -6.52762370e-04,
            #   -1.56835717e-03, -1.03905303e-03, -8.61215014e-04, -1.03753759e-03,
            #    2.40262615e+00])
            x_opt = np.array(x0)
        else:
            assert x0 is not None
            print('TEST doing acutal minimization')
            result = minimize(error, x0, method='Nelder-Mead', options={
                'fatol': 0,
                'maxfev': 10**3
            })
            x_opt = result.x

        print('minimization took', time.time() - start, 'error', error(x_opt))
        self.x_opt = x_opt
        print(x0)
        print(x_opt)
        print()


        if False:
            from fixture.plot_helper import plt

            # TODO this is a temporary plot
            predict_vec = np.vectorize(self.predict, signature='(n),(m)->()')
            predictions = predict_vec(my_data, x0)
            x_label = my_data.columns[0]
            plt.title('TEMP looking at x_init')
            plt.scatter(my_data[x_label], result_data)
            plt.scatter(my_data[x_label], predictions)
            plt.grid()
            plt.show()

            predict_vec = np.vectorize(self.predict, signature='(n),(m)->()')
            predictions = predict_vec(my_data, x_opt)
            x_label = my_data.columns[0]
            plt.scatter(my_data[x_label], result_data)
            plt.scatter(my_data[x_label], predictions)
            plt.grid()
            plt.show()


        return x_opt

    def eval(self, data):
        # for use after fitting has been done
        assert isinstance(data, (dict, pandas.DataFrame))
        assert self.x_opt is not None
        print()
        input_data = [data[input] for input in self.input_signals]
        result = self.predict(input_data, self.x_opt)
        return result

    @abstractmethod
    def verilog(self, opt_names, coef_names):
        # return a string that is a verilog implementation of predict()
        pass


class AnalogExpression(Expression):
    NUM_COEFFICIENTS = 2
    last_coef_is_offset = True

    def __init__(self, opt_signal, name):
        self.name = name
        self.input_signals = [opt_signal]

    def predict(self, opt_values, coefficients):
        assert len(opt_values) == len(self.input_signals)
        assert len(coefficients) == 2
        return opt_values[0] * coefficients[0] + coefficients[1]


class ConstExpression(Expression):
    NUM_COEFFICIENTS = 1
    input_signals = []
    last_coef_is_offset = True

    def predict(self, opt_values, coefficients):
        assert len(opt_values) == 0
        assert len(coefficients) == 1
        return coefficients[0]

class AffineExpression(Expression):
    __doc__ = '''The constructor takes in a list of inputs. A coefficient will 
                 be assigned to each one. There is a constant at the end. '''
    last_coef_is_offset = True

    def __init__(self, inputs, name):
        assert False, 'AffineExpression is still a work in progress'
        self.name = name
        self.input_signals = inputs
        self.NUM_COEFFICIENTS = len(inputs) + 1

    def predict(self, opt_values, coefs):
        # opt_values are essentially input_values
        assert len(opt_values) == len(self.input_signals)
        assert len(coefs) == len(opt_values)
        return sum(o*c for o, c in zip(opt_values, coefs[:-1])) + coefs[-1]

    #def get_x_init(self, optional_data, result_data):
    #    # don't use nonlinear optimizer if you are linear!
    #    abc
    def fit(self, optional_data, result_data):
        print('using linear regression')
        # we want to get a rough fit before we go to the solver
        # we can guess that the bits are thermometer or binary up/down
        # If we know the ratios between bits, we can do linear regression
        opt_values = np.array([optional_data[s] for s in self.input_signals])
        result = np.linalg.pinv(opt_values.T) @ result_data
        self.x_opt = result
        assert False, 'offset?'
        return self.x_opt


class LinearExpression(Expression):
    __doc__ = '''The constructor takes in a list of inputs. A coefficient will 
                 be assigned to each one. There is NO constant at the end. '''
    last_coef_is_offset = False

    def __init__(self, inputs, name):
        self.name = name
        self.input_signals = inputs
        # coefficients do NOT include any constant offset, because often the
        # last input is already a constant
        self.NUM_COEFFICIENTS = len(inputs)

    def predict(self, opt_values, coefs):
        # opt_values are essentially input_values
        assert len(opt_values) == len(self.input_signals)
        assert len(coefs) == len(opt_values)
        return sum(o*c for o, c in zip(opt_values, coefs))

    #def get_x_init(self, optional_data, result_data):
    #    # don't use nonlinear optimizer if you are linear!
    #    abc
    def fit(self, optional_data, result_data):
        print('using linear regression')
        # we want to get a rough fit before we go to the solver
        # we can guess that the bits are thermometer or binary up/down
        # If we know the ratios between bits, we can do linear regression
        opt_values = np.array([optional_data[s] for s in self.input_signals])
        result = np.linalg.pinv(opt_values.T) @ result_data
        self.x_opt = result
        return self.x_opt


class HeirarchicalExpression(Expression):
    def __init__(self, parent_expression, child_expressions, name):
        # each coefficient in the parent expression is the result of evaluating
        # a child expression
        # child_expression_names should match the parent inputs
        self.name = name
        self.manage_offsets = isinstance(parent_expression, SumExpression)



        assert parent_expression.NUM_COEFFICIENTS == (len(child_expressions) + (1 if self.manage_offsets else 0))
        self.parent_expression = parent_expression
        self.child_expressions = child_expressions

        child_inputs = [s for ce in child_expressions for s in ce.input_signals]
        self.input_signals = list(parent_expression.input_signals) + child_inputs

        num_child_coefficients = sum(ce.NUM_COEFFICIENTS for ce in child_expressions)
        # don't add num parent coefficients because those are the child expressions
        if self.manage_offsets:
            # we do something special here by consolidating child offsets
            num_children_offsets = 0
            for child in self.child_expressions:
                if child.last_coef_is_offset:
                    num_children_offsets += 1
            self.num_redundant_offsets = num_children_offsets-1 if num_children_offsets > 0 else 0
            self.NUM_COEFFICIENTS = num_child_coefficients - self.num_redundant_offsets
        else:
            self.num_redundant_offsets = 0
            self.NUM_COEFFICIENTS = num_child_coefficients

    ## anywhere this was used in the past should be replaced by using fit
    ## directly on the child
    #def fit(self, optional_data, result_data):
    #    if (len(self.child_expressions) == 2
    #        and isinstance(self.parent_expression, SumExpression)
    #        and isinstance(self.child_expressions[1], ConstExpression)):
    #        # if a child has a special fit method, we should use it
    #        if hasattr(self.child_expressions[0], 'get_x_init'):
    #            assert False, "I am trying to get rid of get_x_init"
    #            coefs_all = self.child_expressions[0].get_x_init(optional_data, result_data)
    #            self.child_expressions[0].x_init = coefs_all[:-1]
    #            self.child_expressions[1].x_init = coefs_all[-1:]
    #            self.x_init = coefs_all

    #    return super().fit(optional_data, result_data)

    def fit_by_group(self, optional_data, result_data):
        # relies on a specific structure of heirarchy and specific sweep groups
        # in optional_data in order to fit children and grandchildren one
        # piece at a time
        groups = {sg for sg in optional_data[SampleManager.GROUP_ID_TAG]
                  if sg is not None}
        groups = sorted(groups, key=lambda sg: sg.name)

        # first thing is to figure out which sweep groups to use for which
        # fit expressions
        fits_for_sweeps = defaultdict(list)
        for i in range(len(self.child_expressions)):
            child = self.child_expressions[i]
            for grandchild in child.child_expressions:
                if isinstance(grandchild, ConstExpression):
                    # TODO might remove all ConstExpressions in the future
                    # TODO once this is working, remove this whole if statement
                    #  so that users can use ConstExpression
                    assert False, 'should not be using this for now'
                    continue
                # which sweep is best for grandchild.input_signals
                # first, search for an exact match
                def break_buses(ss):
                    # break sample groups that are buses into their bits
                    sg_signals = []
                    for s in ss:
                        if isinstance(s, SignalArray):
                            assert len(s.array.shape) == 1, 'TODO nested buses'
                            sg_signals += list(s)
                        else:
                            sg_signals.append(s)
                    return sg_signals
                perfect_match_found = False
                for sg in groups:
                    if (set(break_buses(grandchild.input_signals))
                            == set(break_buses(sg.signals))):
                        # perfect match, use this sweep group
                        fits_for_sweeps[sg].append(grandchild)
                        perfect_match_found = True

                # TODO this will include all partial matches, but in reality
                #  I think we should rank them and include only the best
                if not perfect_match_found:
                    # no perfect match found
                    # partial match is when the sweep group has extra things
                    for sg in groups:
                        if all(s in break_buses(sg.signals)
                               for s in break_buses(grandchild.input_signals)):
                            # all the things in the fit expr were swept
                            fits_for_sweeps[sg].append(grandchild)

        # now we loop through each SampleGroup
        result_fits = defaultdict(list)
        for sg in groups:
            # group_data is everything during the sweep of sg
            group_data_indices = optional_data[SampleManager.GROUP_ID_TAG] == sg
            group_data = optional_data[group_data_indices]
            group_data_res = result_data[group_data_indices]

            # First, we find values for each param, with optional inputs fixed
            sweep_ids = {si for si in group_data[SampleManager.SWEEP_ID_TAG]}
            group_results = []
            # example rows are one row from each sweep point
            example_rows = []
            plot_xs = []
            plot_ys = []
            plot_predictions = []
            for si in sweep_ids:
                # point_data is the data from one point (si) on the sg sweep
                point_indices = group_data[SampleManager.SWEEP_ID_TAG] == si
                point_data = group_data[point_indices]
                example_rows.append(point_indices.idxmax())
                point_data_res = group_data_res[point_indices]
                self.parent_expression.fit(point_data, point_data_res)
                group_results.append(self.parent_expression.x_opt)

                # TODO in general this plot xaxis is not right
                xaxis = self.parent_expression.input_signals[0]
                plot_xs.append(point_data[xaxis])
                plot_ys.append(point_data_res)
                predict_data = [point_data[input] for input in self.parent_expression.input_signals]
                predictions = self.parent_expression.predict(predict_data, self.parent_expression.x_opt)
                plot_predictions.append(predictions)
            group_results = np.array(group_results)

            if PLOT:
                # plotting initial small plots
                print('New figure for ', sg)
                plt.figure()
                for xs, ys, preds in zip(plot_xs, plot_ys, plot_predictions):
                    plt.plot(xs, ys, '*')
                    plt.plot(xs, preds, '--')
                plt.legend([f'id{i}' for i in sweep_ids])
                plt.title(f'Results from sweeping {sg}')
                plt.xlabel(f'{xaxis}')
                plt.ylabel(f'f{self.parent_expression.name}')
                plt.grid()
                PlotHelper.save_current_plot(f'Fits for {self.name} from Sweeping {sg.name}')

            # Next we find expressions for each param, using values from earlier
            # each row in example_data corresponds to one point in the sweep
            # TODO I'd like to delete the columns corresponding to test inputs
            #  in example_data because they are not relevant. But I don't have
            #  a good way to find the names for those
            example_data = group_data.loc[example_rows]
            for i in range(len(self.child_expressions)):
                child = self.child_expressions[i]
                child_results = group_results[:, i]

                # When we fit by group we don't exactly follow the normal
                # heirarchy for the child, so it has to be in a specific form
                assert isinstance(child, HeirarchicalExpression)
                assert isinstance(child.parent_expression, SumExpression)
                # No longer need this last one since children have offsets now
                #assert isinstance(child.child_expressions[-1], ConstExpression)

                ## now we look through child's children and determine which
                # one(s) are actually affected by changing sg
                # TODO sg.signal doesn't always exist?
                # TODO what about when sg has multiple signals?
                # TODO what about when child.input_signals has multiple signals?
                #relevant_grandchildren = []
                for grandchild in child.child_expressions:
                    # TODO delete old strategy
                    #if any(s in grandchild.input_signals for s in sg.signals):
                    #    relevant_grandchildren.append(grandchild)
                    #elif (isinstance(sg.signal, SignalArray)
                    #      and all(s in grandchild.input_signals for s in sg.signal)):
                    #    relevant_grandchildren.append(grandchild)
                    if grandchild not in fits_for_sweeps[sg]:
                        continue

                    # No longer need to create a dummy expression to add const
                    #expression_list = [grandchild]
                    ## add constant
                    #expression_list.append(child.child_expressions[-1])
                    #temp_expression = HeirarchicalExpression(
                    #    SumExpression(len(expression_list), 'temp_sum'),
                    #    expression_list,
                    #    'temp_expr')
                    #temp_expression.fit(example_data, child_results)
                    grandchild.fit(example_data, child_results)

                    if PLOT:
                        # plotting secondary fits, from using fit params as goals
                        plt.figure()
                        # TODO this may not work with future sg types
                        xaxis = sg.signals[-1]
                        xs = example_data[xaxis]
                        #xs_sampler = fixture.sampler.get_sampler_for_signal(xaxis)
                        x_targets = np.linspace(0, 1, 100).reshape((100,1))
                        targets = np.concatenate((x_targets, 0.5*np.ones((100, sg.NUM_DIMS-1))), 1)
                        data_smooth = pandas.DataFrame(sg.get_many(targets))
                        data_smooth = data_smooth.sort_values(by=[xaxis])
                        predictions = grandchild.predict_from_dict(data_smooth, grandchild.x_opt)
                        xs_smooth = data_smooth[xaxis]

                        xs_order = np.argsort(xs)
                        plt.plot(np.array(xs)[xs_order], child_results[xs_order], '*')
                        plt.plot(xs_smooth, predictions, 'x--')
                        plt.xlabel(f'{xaxis}')
                        plt.ylabel(f'{child.name}')
                        plt.grid()
                        PlotHelper.save_current_plot(f'Initial fit for {grandchild.name} to {self.name} vs {sg}')

                        # TEMP for checking x_init
                        if grandchild.x_init is not None:
                            predictions = grandchild.predict_from_dict(data_smooth, grandchild.x_init)
                            #plt.plot(xs, child_results, '*')
                            plt.plot(np.array(xs)[xs_order],
                                     child_results[xs_order], '*')
                            plt.plot(xs_smooth, predictions, 'x--')
                            plt.xlabel(f'{xaxis}')
                            plt.ylabel(f'{child.name}')
                            plt.grid()
                            PlotHelper.save_current_plot(f'Debug x_init for {grandchild.name} to {self.name} vs {sg}')
                        print()


                    result_fits[grandchild].append(grandchild.x_opt)

        # I think that result_fits should have one entry per optional input
        # expression, plus one entry per optional input for the constants
        # With proper centering, the const fits should all be equal, so we
        # just use their average
        # TODO I'm not sure it's wise to directly edit grandchild.x_init,
        #  but I think since the objects won't be used to fit again it's
        #  probably okay
        # TODO redundant offsets?
        for grandchild, fits in result_fits.items():
            x_init = sum(fits) / len(fits)
            grandchild.x_init = x_init

        # now we are ready
        self.fit(optional_data, result_data)

        if PLOT:
            predict_data = [optional_data[col] for col in self.input_signals]
            predictions = self.predict(predict_data, self.x_opt)
            # TODO fix this xaxis
            xaxis = self.parent_expression.input_signals[0]
            xs = optional_data[xaxis]
            plt.figure()
            plt.plot(xs, result_data, '*')
            plt.plot(xs, predictions, '--')
            plt.grid()
            plt.xlabel(f'{xaxis}')
            plt.ylabel(f'{self.name}')
        print()

    def predict(self, opt_values, coefs):
        # first, use the child expressions to find each param
        assert len(coefs) == self.NUM_COEFFICIENTS
        assert len(opt_values) == len(self.input_signals)
        coef_count = 0
        parent_coefs = []
        for ce in self.child_expressions:
            input_indices_ce = [self.input_signals.index(s) for s in ce.input_signals]
            input_values_ce = [opt_values[i] for i in input_indices_ce]

            # take the next slice of coefs for ce, but if ce has a redundant
            # offset then we should just pass it a zero for that offset instead
            # num_child_coefficients is the number of our coefficients to slice
            # out for this child, so it does NOT include the offset
            redundant_offset = self.manage_offsets and ce.last_coef_is_offset
            num_child_coefficients = ce.NUM_COEFFICIENTS
            if redundant_offset:
                num_child_coefficients -= 1
            coefs_ce = coefs[coef_count : coef_count + num_child_coefficients]
            coef_count += num_child_coefficients
            if redundant_offset:
                coefs_ce = np.concatenate((coefs_ce, [0]))

            parent_coef = ce.predict(input_values_ce, coefs_ce)
            parent_coefs.append(parent_coef)

        if self.manage_offsets:
            # should be 1 left over for the aggregate offset
            parent_coefs.append(coefs[coef_count])
            coef_count += 1

        assert coef_count == len(coefs)

        parent_input_indices = [self.input_signals.index(s) for s in
                            self.parent_expression.input_signals]
        parent_input_values = [opt_values[i] for i in parent_input_indices]
        ans = self.parent_expression.predict(parent_input_values, parent_coefs)
        return ans


    @property
    def x_opt(self):
        return self._x_opt
    @x_opt.setter
    def x_opt(self, x_opt):
        # set self.opt for self and all the children
        self._x_opt = x_opt
        x_opt_count = 0
        for ce in self.child_expressions:
            slice_length = ce.NUM_COEFFICIENTS
            if self.manage_offsets and ce.last_coef_is_offset:
                slice_length -= 1
            x_opt_ce = x_opt[x_opt_count : x_opt_count + slice_length]
            if self.manage_offsets and ce.last_coef_is_offset:
                # TODO maybe choose offset so influence at nominal is zero
                x_opt_ce = np.concatenate((x_opt_ce, [0]))
            x_opt_count += slice_length
            ce.x_opt = x_opt_ce

        if self.manage_offsets:
            # don't do anything with managed offset x_opt[-1]
            x_opt_count += 1

        assert x_opt_count == len(x_opt)

    @property
    def x_init(self):
        # we need to aggregate x_init from children, taking care with
        # redundant offsets
        x_init = []
        aggregate_offset = 0
        for ce in self.child_expressions:
            ce_x_init = ce.x_init if ce.x_init is not None else np.ones(ce.NUM_COEFFICIENTS)
            if self.manage_offsets and ce.last_coef_is_offset:
                x_init = np.concatenate((x_init, ce_x_init[:-1]))
                aggregate_offset += ce_x_init[-1]
            else:
                x_init = np.concatenate((x_init, ce_x_init))

        if self.manage_offsets:
            x_init = np.concatenate((x_init, [aggregate_offset]))

        return x_init
    @x_init.setter
    def x_init(self, x_init):
        raise NotImplementedError('Cannot set heirarchical x_init directly')

    def _search(self, target_name):
        # return a list of child expressions with the target name
        matches = []
        if self.name == target_name:
            matches.append(self)

        for child in self.child_expressions:
            if isinstance(child, HeirarchicalExpression):
                matches += child._search(target_name)
            else:
                if child.name == target_name:
                    matches.append(child)
        return matches

    def search(self, target_name):
        # return a child expression with the target_name, or None
        matches = self._search(target_name)
        if len(matches) == 0:
            return None
        elif len(matches) > 1:
            raise KeyError(f'Found multiple Expressions matching "{target_name}" in "{self.name}"')
        else:
            return matches[0]

class SumExpression(Expression):
    input_signals = []

    def __init__(self, n, name):
        self.name = name
        self.NUM_COEFFICIENTS = n

    def predict(self, opt_values, coefs):
        assert len(opt_values) == 0
        return sum(coefs)


class ReciprocalExpression(Expression):
    __doc__ = '''
The constructor takes in a list of inputs. This is intended to model the
resistance of a bank of resistors. 
1/(ctrl[0]/R0 + ctrl[1]/R1 + ... + 1/Rnom)
Issue is that this expression is difficult to zero out for parameters that don't
depend on R. An alternative is:
Rnom/(ctrl[0]/X0 + ctrl[1]/X1 + ... + 1)
Issue with that one is that it's difficult if the crtl=0 resistance is 0.
We end up going with this weird version:
G/(ctrl[0] + ctrl[1]/X1 + ... + Y)
I still consider this a work-in-progress. I think it makes sense if you 
interpret G as a sort of "global conductance multiplier" for each bit
1/(ctrl[0]/Rnom + ctrl[1]/(Rnom*X1) + ctrl[2]/(Rnom*X2) + ... + Y/Rnom)
Computationally, we pull Rnom on the top to avoid precision weirdness
Note that Y is on the top of that last term because Y=0 is common, Y=inf is not
For ctrl[5:0], coefs = [Rnom, X1, X2, X3, X4, X5, Y]
Also, there is a additional additive offset at the end
For ctrl[5:0], coefs = [Rnom, X1, X2, X3, X4, X5, Y, offset]
'''
    last_coef_is_offset = True

    def __init__(self, inputs, name):
        self.name = name
        self.input_signals = inputs
        self.NUM_COEFFICIENTS = len(inputs) + 2
        self.x_init = np.ones(self.NUM_COEFFICIENTS)

        ## TODO get rid of this
        #self.x_init[0] = 1500
        #self.x_init[1] = 2
        #self.x_init[2] = 4
        #self.x_init[3] = 8
        #self.x_init[4] = 8
        #self.x_init[5] = 8
        #self.x_init[6] = 0


    def predict(self, opt_values, coefs):
        # opt_values are essentially input_values
        assert len(opt_values) == len(self.input_signals)
        assert len(coefs) == len(opt_values) + 2
        denominator = opt_values[0] + sum(o / c for o, c in zip(opt_values[1:], coefs[1:-2])) + coefs[-2]
        return coefs[0] / denominator + coefs[-1]

    def get_x_init(self, optional_data, result_data):
        # we want to get a rough fit before we go to the solver
        # we can guess that the bits are thermometer or binary up/down
        # If we know the ratios between bits, we can do linear regression
        assert False, 'trying to phase this out, right?'

    def fit(self, optional_data, result_data):
        def quick_fit(xs, ys):
            result = linregress(xs, ys)
            return result.rvalue**2, result.slope, result.intercept
        bits = [optional_data[s] for s in self.input_signals]
        denom_coefss = []
        therm = [1 for i in range(1, len(bits))] + [0]
        denom_coefss.append(therm)
        binary_inc = [2**i for i in range(1, len(bits))] + [1e-3]
        denom_coefss.append(binary_inc)
        binary_dec = [(1/2)**i for i in range(1, len(bits))] + [1e-3]
        denom_coefss.append(binary_dec)

        # r2_value, slope, intercept, denom_coefs
        # r2_value is first so we can use comparison to find the highest
        fit = (0.0, 0, 0, None)
        for denom_coefs in denom_coefss:
            denom = sum(b/c for b, c in zip(bits, [1]+denom_coefs[:-1])) + denom_coefs[-1]
            xs = 1/denom
            result = quick_fit(xs, result_data)
            fit = max(fit, (*result, denom_coefs))
        r2_value, slope, intercept, denom_coefs = fit

        coefs = [slope] + denom_coefs + [intercept]
        self.x_init = coefs
        return super().fit(optional_data, result_data)


def get_optional_expression_from_signal(s, name):
    if isinstance(s, SignalArray):
        print('TODO fix SignalArray behavior')
        #return LinearExpression(list(s), name)
        return ReciprocalExpression(list(s), name)

    else:
        assert isinstance(s, SignalIn)
        assert s.type_ == 'analog', 'TODO'
        assert len(s.value) == 2
        return AnalogExpression(s, name)


def get_optional_expression_from_signals(s_list, name):
    assert isinstance(name, str)
    individual = []
    for s in s_list:
        child_name = f'{name}_{s.friendly_name()}'
        individual.append(get_optional_expression_from_signal(s, child_name))
    # TODO delete old ConstExpression
    #individual.append(ConstExpression(f'{name}_const'))
    # +1 is for a nominal offset
    combiner = SumExpression(len(individual)+1, f'{name}_summer')
    total = HeirarchicalExpression(combiner, individual, name)
    return total



def test_optimizers(error_fun, N_COEFS):
    guesses = []

    # zeros
    x0 = np.zeros(N_COEFS)
    x0[6] = 1
    x0[14] = 1
    x0[22] = 1
    guesses.append(x0)

    # ones
    x0 = np.ones(N_COEFS)
    guesses.append(x0)

    # good
    x0 = np.zeros(N_COEFS)
    for i in range(3):
        # bottom of fraction
        for j in range(7):
            x0[8*i+j] = 1e-3
        # constant offset
        x0[8*i+7] = 0
    guesses.append(x0)

    # great
    # essentially null out cm and const terms
    x0 = np.zeros(N_COEFS)
    x0[14] = 1e6
    x0[22] = 1e6
    for i in range(6):
       resistance = 2**i * 1e3
       x0[i] = 1/resistance
    # last one should be zero, but I'm afraid of divide by zero issues
    x0[6] = 1e-7
    guesses.append(x0)


    import time
    for i, x0 in enumerate(guesses):
        print('round', i, 'guess', x0)
        start = time.time()

        result = minimize(error_fun, x0, method='Powell', options={
            #'gtol': 0,
            'ftol': 0, # Powell
            #'fatol': 0, # Nelder-Mead
            'maxfev': 5*10 ** 3,
            'disp': True
        })
        x_opt = result.x

        print('minimization took', time.time() - start)
        print('result', x_opt)
        print('nfev', result.nfev, 'success', result.success)
        e = error_fun(x_opt)
        print('error', e)
        print()
    print()
