from abc import ABC, abstractmethod
from collections import defaultdict

from scipy.optimize import minimize
import numpy as np
import pandas

from fixture.sampler import SampleManager
from fixture.signals import SignalArray, SignalIn

PLOT = False

class Expression:
    x_init = None
    x_opt = None

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

    @abstractmethod
    def predict(self, opt_values, coefs):
        # given the values of the optional inputs and the fit coefficients,
        # calculate the linear influence of the optional inputs
        pass

    def fit(self, optional_data, result_data):
        # return a best-fit of the coefficients, i.e. minimize
        # predict(optional_data, coefficientss) - result_data
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

        def get_x_init(exp):
            if isinstance(exp, HeirarchicalExpression):
                return np.concatenate([get_x_init(e)
                                 for e in exp.child_expressions])
            else:
                if exp.x_init is None:
                    return np.ones(exp.NUM_COEFFICIENTS)
                else:
                    assert len(exp.x_init) == exp.NUM_COEFFICIENTS
                    return exp.x_init
        x0 = get_x_init(self)

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
        if False or self.name == 'out_diff':
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


        elif False or self.name == 'out_cm':
            #x_opt = np.array([-2.19233927e+01,  2.61691713e+02,  1.38773168e+01,  1.57116746e+02,
            #    4.38110640e+02,  3.03643386e+01,  1.84852178e+02,  6.07208190e+02,
            #    6.80619246e+02,  7.28624576e+02,  6.60212312e+02,  6.29206962e+02,
            #    7.36672174e+02,  1.23052961e+03, -1.14858118e-03, -6.52762370e-04,
            #   -1.56835717e-03, -1.03905303e-03, -8.61215014e-04, -1.03753759e-03,
            #    2.40262615e+00])
            x_opt = np.array(x0)
        else:
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
    NUM_COEFFICIENTS = 1

    def __init__(self, opt_signal, name):
        self.name = name
        self.input_signals = [opt_signal]

    def predict(self, opt_values, coefficients):
        assert len(opt_values) == len(self.input_signals)
        assert len(coefficients) == 1
        return opt_values[0] * coefficients[0]


class ConstExpression(Expression):
    NUM_COEFFICIENTS = 1
    input_signals = []

    def predict(self, opt_values, coefficients):
        assert len(opt_values) == 0
        assert len(coefficients) == 1
        return coefficients[0]


class LinearExpression(Expression):
    __doc__ = '''The constructor takes in a list of inputs. A coefficient will 
                 be assigned to each one. There is no constant added. '''
    def __init__(self, inputs, name):
        self.name = name
        self.input_signals = inputs
        # coefficients do NOT include any constant offset, because that is
        # handled separately by the tool
        self.NUM_COEFFICIENTS = len(inputs)

    def predict(self, opt_values, coefs):
        # opt_values are essentially input_values
        assert len(opt_values) == len(self.input_signals)
        assert len(coefs) == len(opt_values)
        return sum(o*c for o, c in zip(opt_values, coefs))


class HeirarchicalExpression(Expression):
    def __init__(self, parent_expression, child_expressions, name):
        # each coefficient in the parent expression is the result of evaluating
        # a child expression
        # child_expression_names should match the parent inputs
        self.name = name
        assert parent_expression.NUM_COEFFICIENTS == len(child_expressions)
        self.parent_expression = parent_expression
        self.child_expressions = child_expressions

        child_inputs = [s for ce in child_expressions for s in ce.input_signals]
        self.input_signals = list(parent_expression.input_signals) + child_inputs

        num_child_coefficients = sum(ce.NUM_COEFFICIENTS for ce in child_expressions)
        # don't add num parent coefficients because those are the child expressions
        self.NUM_COEFFICIENTS = num_child_coefficients

    #def fit(self, optional_data, result_data, swept_group=None):
    #    # can we detect when we have the data to fit children individually?
    #    # I think we actually need the whole result dataframe so we can read
    #    # the tags instead of trying to figure out what's changing
    #    for child in self.child_expressions:
    #        pass
    #    assert False, 'todo'

    def fit_by_group(self, optional_data, result_data):
        groups = {sg for sg in optional_data[SampleManager.GROUP_ID_TAG]
                  if sg is not None}
        groups = sorted(groups, key=lambda sg: sg.name)
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

                xaxis = self.parent_expression.input_signals[0]
                plot_xs.append(point_data[xaxis])
                plot_ys.append(point_data_res)
                predict_data = [point_data[input] for input in self.parent_expression.input_signals]
                predictions = self.parent_expression.predict(predict_data, self.parent_expression.x_opt)
                plot_predictions.append(predictions)
            group_results = np.array(group_results)

            if PLOT:
                # plotting initial small plots
                from fixture.plot_helper import plt
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
                plt.show()

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
                assert isinstance(child.child_expressions[-1], ConstExpression)

                # now we look through child's children and determine which
                # one(s) are actually affected by changing sg
                # TODO sg.signal doen't always exist?
                # TODO what about when sg has multiple signals?
                # TODO what about when child.input_signals has multiple signals?
                relevant_grandchildren = []
                for grandchild in child.child_expressions:
                    if sg.signal in grandchild.input_signals:
                        relevant_grandchildren.append(grandchild)
                    elif (isinstance(sg.signal, SignalArray)
                          and all(s in grandchild.input_signals for s in sg.signal)):
                        relevant_grandchildren.append(grandchild)
                relevant_grandchildren.append(child.child_expressions[-1])
                temp_expression = HeirarchicalExpression(
                    SumExpression(len(relevant_grandchildren), 'temp_sum'),
                    relevant_grandchildren,
                    'temp_expr')
                temp_expression.fit(example_data, child_results)

                if PLOT:
                    # plotting secondary fits, from using fit params as goals
                    from fixture.plot_helper import plt
                    plt.figure()
                    predict_data = [example_data[s] for s in temp_expression.input_signals]
                    predictions = temp_expression.predict(predict_data, temp_expression.x_opt)
                    # TODO this may not work with future sg types
                    xaxis = sg.signal
                    xs = example_data[xaxis]
                    plt.plot(xs, child_results, '*')
                    plt.plot(xs, predictions, '--')
                    plt.xlabel(f'{xaxis}')
                    plt.ylabel(f'{child.name}')
                    plt.grid()
                    #plt.show()


                for grandchild in temp_expression.child_expressions:
                    result_fits[grandchild].append(grandchild.x_opt)

        # I think that result_fits should have one entry per optional input
        # expression, plus one entry per optional input for the constants
        # With proper centering, the const fits should all be equal, so we
        # just use their average
        # TODO I'm not sure it's wise to directly edit grandchild.x_init,
        #  but I think since the objects won't be used to fit again it's
        #  probably okay
        for grandchild, fits in result_fits.items():
            x_init = sum(fits) / len(fits)
            grandchild.x_init = x_init

        # now we are ready
        self.fit(optional_data, result_data)
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
            coefs_ce = coefs[coef_count : coef_count + ce.NUM_COEFFICIENTS]
            coef_count += ce.NUM_COEFFICIENTS

            parent_coef = ce.predict(input_values_ce, coefs_ce)
            parent_coefs.append(parent_coef)

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
            x_opt_ce = x_opt[x_opt_count : x_opt_count + ce.NUM_COEFFICIENTS]
            x_opt_count += ce.NUM_COEFFICIENTS
            ce.x_opt = x_opt_ce

        assert x_opt_count == len(x_opt)

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
    __doc__ = '''The constructor takes in a list of inputs. A coefficient will 
                 be assigned to each one, plus a constant, and all that is 
                 put on the bottom of the fraction. '''

    def __init__(self, inputs, name):
        self.name = name
        self.input_signals = inputs
        self.NUM_COEFFICIENTS = len(inputs) + 1
        self.x_init = np.ones(self.NUM_COEFFICIENTS)
        self.x_init[0] = 2000

    def predict(self, opt_values, coefs):
        # opt_values are essentially input_values
        assert len(opt_values) == len(self.input_signals)
        assert len(coefs) == len(opt_values) + 1
        denominator = sum(o / c for o, c in zip(opt_values, coefs[1:])) + 1
        return coefs[0] / denominator


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
    individual.append(ConstExpression(f'{name}_const'))
    combiner = SumExpression(len(individual), f'{name}_summer')
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

        result = minimize(error_fun, x0, method='Nelder-Mead', options={
            #'gtol': 0,
            #'ftol': 0,
            'fatol': 0,
            'maxfev': 10 ** 3,
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
