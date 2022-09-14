from abc import ABC, abstractmethod
from scipy.optimize import minimize
import numpy as np


class SampleManager:
    def __init__(self, num_params, optional_groups, test_inputs):
        self.optional_groups = optional_groups
        self.test_inputs


    def sweep_one(self, group):
        # group is the one optional group to sweep, while holding others nominal
        # I ran into the issue of how to break this into independent axes
        # I think the solution is to do one orthogonal sample for all the
        # optional group, and then individual orthogonal samples for the
        # test inputs
        # It's possible that I should do all the things in one go and then
        # condense the optional things into groups, but that's difficult when
        # the optional group has multiple dimensions
        TODO

class SampleTODO(ABC):
    @abstractmethod
    def get(self, target):
        # target_samples are between 0 and 1
        # they should be translated to the appropriate space for this input
        pass

    @abstractmethod
    def get_nominal(self):
        # return 1 sample at the nominal value
        pass

    @abstractmethod
    def get_plot_value(self, samples):
        # given a value of the optional input(s), return the value you would
        # want on a plot axis. Usually this is an identity function, but for
        # arrays of bits it would convert to a decimal value
        pass





class SamplerAnalog(SampleTODO):
    def __init__(self, limits):
        assert len(limits) == 2, 'Fit requires an input range'
        self.limits = limits

    def random_samples(self, target_samples):
        pass


# --------------------------------------



class Expression:
    @property
    @abstractmethod
    def NUM_COEFFICIENTS(self):
        # a constant property that is the number of coefs to fit
        # Remember that the constant offset is separate from this count
        pass

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
    def predict(self, opt_values, coefs, offset):
        # given the values of the optional inputs and the fit coefficients,
        # calculate the linear influence of the optional inputs
        pass

    def fit(self, optional_data, result_data):
        # return a best-fit of the coefficients, i.e. minimize
        # predict(optional_data, coefficientss) - result_data
        # return a tuple of (coefficients, offset)
        assert len(optional_data) == len(result_data)
        assert optional_data.shape == (len(result_data), len(self.input_signals))
        def error(coefs):
            predict_vec = np.vectorize(self.predict, signature='(n),(m),()->()')
            predictions = predict_vec(optional_data, coefs[:-1], coefs[-1])
            assert len(result_data) == len(predictions)
            errors = result_data - predictions
            e = sum(errors**2)
            return e

        x0 = np.zeros(self.NUM_COEFFICIENTS+1)
        result = minimize(error, x0)
        x_opt = result.x
        return (x_opt[:-1], x_opt[-1])

    @abstractmethod
    def verilog(self, opt_names, coef_names):
        # return a string that is a verilog implementation of predict()
        pass


class AnalogExpression(Expression):
    NUM_COEFFICIENTS = 1

    def __init__(self, opt_signal):
        self.input_signals = [opt_signal]

    def predict(self, opt_values, coefficients, offset):
        assert len(opt_values) == len(self.input_signals)
        assert len(coefficients) == 1
        return opt_values[0] * coefficients[0]


class LinearExpression(Expression):
    def __init__(self, inputs):
        self.input_signals = inputs
        # coefficients to NOT include any constant offset, but that is added
        # separately as an input to predict()
        self.NUM_COEFFICIENTS = len(inputs)

    def predict(self, opt_values, coefs, offset):
        # opt_values are essentially input_values
        assert len(opt_values) == len(self.input_signals)
        assert len(coefs) == len(opt_values)
        return sum(opt_values * coefs) + offset

class HeirarchicalExpression(Expression):
    def __init__(self, parent_expression, child_expressions):
        # each coefficient in the parent expression is the result of evaluating
        # a child expression
        # child_expression_names should match the parent inputs
        assert parent_expression.NUM_COEFFICIENTS == len(child_expressions)
        self.parent_expression = parent_expression
        self.child_expressions = child_expressions

        child_inputs = [s for ce in child_expressions for s in ce.input_signals]
        self.input_signals = list(parent_expression.input_signals) + child_inputs

        num_child_coefficients = sum(ce.NUM_COEFFICIENTS for ce in child_expressions)
        self.NUM_COEFFICIENTS = num_child_coefficients + parent_expression.NUM_COEFFICIENTS


    def predict(self, opt_values, coefs, offset):
        # first, use the child expressions to find each param
        assert len(coefs) == self.NUM_COEFFICIENTS
        assert len(opt_values) == len(self.input_signals)
        assert offset == 0, 'Specially for param algebra, any offset is included internally'
        coef_count = 0
        parent_coefs = []
        for ce in self.child_expressions:
            input_indices_ce = [self.input_signals.index(s) for s in ce.input_signals]
            input_values_ce = [opt_values[i] for i in input_indices_ce]
            coefs_ce = coefs[coef_count : coef_count + ce.NUM_COEFFICIENTS]
            coef_count += ce.NUM_COEFFICIENTS

            parent_coef = ce.predict(input_values_ce, coefs_ce, offset)
            parent_coefs.append(parent_coef)

        assert coef_count == len(coefs)

        parent_input_indices = [self.input_signals.index(s) for s in
                            self.parent_expression.input_signals]
        parent_input_values = [opt_values[i] for i in parent_input_indices]
        ans = self.parent_expression.predict(parent_input_values, parent_coefs, offset)
        return ans


