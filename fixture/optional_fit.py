from abc import ABC, abstractmethod


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

    @abstractmethod
    def predict(self, opt_value, coefs, offset):
        # given the values of the optional inputs and the fit coefficients,
        # calculate the linear influence of the optional inputs
        pass

    def fit(self, optional_data, result_data):
        # return a best-fit of the coefficients, i.e. minimize
        # predict(optional_data, coefficientss) - result_data
        # return a tuple of (coefficients, offset)
        assert False, 'TODO default implementation'
        pass

    @abstractmethod
    def verilog(self, opt_names, coef_names):
        # return a string that is a verilog implementation of predict()
        pass


class AnalogExpression(Expression):
    NUM_COEFFICIENTS = 1

    def predict(self, opt_value, coefficients, offset):
        assert len(coefficients) == 1
        return opt_value * coefficients[0]


