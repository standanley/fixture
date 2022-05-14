from collections import defaultdict
from functools import reduce

import statsmodels.formula.api as smf
import pandas
from itertools import combinations, product
import magma
import re
from ast import literal_eval

from fixture.signals import SignalArray
import operator


class Regression:
    # statsmodels gets confused if you try to use '1' to mean a column of 
    # ones, so we manually add a column of ones with the following name
    one_literal = 'const_1'

    @classmethod
    def get_optional_pin_expression(cls, template):
        '''
        Given the magma circuit template, look at the optional pins and create a
        (string) R expression for how the analog pins affect something
        '''

        # TODO pull these out of some dict
        analog_order = template.extras.get('analog_order', 1)
        interaction_a_a = False
        interaction_a_ba = False
        interaction_ba_ba = False

        opt_signals = template.signals.optional_expr()
        opt_signals_flat = [s for x in opt_signals for s in (x if isinstance(x, SignalArray) else [x])]
        # TODO accept real in addition to analog?
        opt_a = [s for s in opt_signals_flat if s.type_ == 'analog']
        opt_ba = [s for s in opt_signals_flat if s.type_ == 'binary_analog']

        terms = [cls.one_literal]
        for a in opt_a:
            for i in range(1, analog_order + 1):
                if i == 1:
                    terms.append(a)
                else:
                    terms.append(tuple([a]*i))

        for ba in opt_ba:
            terms.append(ba)


        def interact(iterator):
            for a, b in iterator:
                terms.append((a, b))

        if interaction_a_a:
            interact(combinations(opt_a, 2))
        if interaction_ba_ba:
            interact(combinations(opt_ba, 2))
        if interaction_a_ba:
            interact(product(opt_a, opt_ba))

        return terms

    @staticmethod
    def clean_string(s):
        # discrepency between the way spice and magma do braces
        # also Patsy won't use < in a variable name
        temp = s.replace('<', '_').replace('>', '_')
        final = temp.replace('[', '_').replace(']', '_')
        return final

    @staticmethod
    def regression_name(s):
        if type(s) == str:
            return Regression.clean_string(s)
        if s.template_name is not None:
            return Regression.clean_string(s.template_name)
        else:
            assert s.spice_name is not None, f'Signal {s} has neither template nor spice name!'
            return Regression.clean_string(s.spice_name)

    @staticmethod
    def vector_parameter_name_output(name, vec_i, signal):
        return f'{name}_outvec[{vec_i}]'

    @staticmethod
    def vector_parameter_name_input(name, vec_i, signal):
        return f'{name}_invec[{vec_i}]'

    @staticmethod
    def vector_input_name(name, vec_i):
        return f'{name}_invec[{vec_i}]'

    def get_terms(self, rhs, opt):
        # essentially an outer product of rhs and opt
        # we delete ones so a const one is just an empty tuple
        # For each term we need
        # 1) unique name (can assume products are unique)
        # 2) tuple of product components (for calculating value)
        # 3) Which param it is part of
        # 4) The optional pin it corresponds to (for reporting to user)
        ones = ['1', self.one_literal]
        ones = ones + [(x,) for x in ones]
        def product(a, b):
            if a in ones:
                a = tuple()
            if b in ones:
                b = tuple()
            if isinstance(a, tuple):
                if isinstance(b, tuple):
                    return (*a, *b)
                else:
                    return (*a, b)
            else:
                if isinstance(b, tuple):
                    return (a, *b)
                else:
                    return (a, b)
        ans = []
        for param, a_term in rhs.items():
            for b_term in opt:
                term = product(a_term, b_term)
                name = self.name_term(term)
                ans.append((name, term, param, b_term))
        return ans

    def name_term(self, term):
        assert isinstance(term, tuple)
        if term == tuple():
            return self.one_literal

        return '_times_'.join([self.regression_name(x) for x in term])

    def __init__(self, template, test, data):
        '''
        Incoming data should be of the form 
        {pin:[x1, ...], ...}
        '''

        assert self.one_literal in data, 'Should have been set in go()'

        # look for optional outputs
        # don't edit the parameter_algebra directly, could persist to another
        # call to fixture.run
        pa = test.parameter_algebra_vectored.copy()
        for s in test.signals.auto_measure():
            if s.spice_name in data.columns:
                # add parameter_algebra entry
                pa[s.spice_name] = {f'{s.spice_name}_meas': '1'}
            else:
                assert False, f'Where to get value for auto_measure {s}?'

        self.consts = {}

        '''
        The format for results is:
        results[lhs][param][opt] = coef
        example:
        {'amp_output':
            {'dcgain':
                 {'const_1': 0.2658061976765506,
                  <vdd>: -0.011425713964578879},
             }
        }
        '''
        results = {lhs: defaultdict(dict) for lhs in pa}
        results_models = {}
        regression_dicts = []
        for lhs, rhs in pa.items():
            lhs_clean = self.clean_string(lhs)
            regression_data_dict = {lhs_clean: data[lhs]}
            self.info_mapping = {}

            optional_pin_expr = self.get_optional_pin_expression(template)
            rhs_info = self.get_terms(rhs, optional_pin_expr)
            for name, term, param, opt in rhs_info:
                assert name not in self.info_mapping, f'Duplicate term: {name}'
                self.info_mapping[name] = (term, param, opt)
                column = reduce(operator.mul,
                                [data[x] for x in term],
                                data[self.one_literal])
                regression_data_dict[name] = column
            # TODO if every instance of 'vdd' is in a product term, should we
            # still add 'vdd' to the dataframe by itself?
            regression_data = pandas.DataFrame(regression_data_dict)


            df_row_mask = ~ regression_data[lhs_clean].isnull()
            df_filtered = regression_data[df_row_mask]

            formula = self.make_formula(lhs_clean, [x[0] for x in rhs_info])
            stats_model = smf.ols(formula, df_filtered)
            stat_results = stats_model.fit()
            results_entry = results[lhs]
            for name, coef in stat_results.params.items():
                term, param, opt = self.info_mapping[name]
                assert opt not in results_entry[param], f'Parameter {lhs}->{param}->{opt} found in multiple parameter algebra formulas'
                results_entry[param][opt] = coef

            results_models[lhs] = stat_results

            # TODO need to double-check that this still handles NaNs correctly
            regression_dicts.append(regression_data)


        # combine the individual dataframes into a big one
        # when they have the same column heading, assert that the data is equal
        data_combined = {}
        for df in regression_dicts:
            for header, column in df.items():
                if header in data_combined:
                    # duplicate
                    assert all(column == data_combined[header]), 'Mismatched data'
                else:
                    data_combined[header] = column
        df_combined = pandas.DataFrame(data_combined)

        # TODO dump res to a yaml file
        self.regression_dataframe = df_combined
        self.results = results
        self.results_models = results_models


    def make_formula(self, lhs_clean, rhs_names):
        return f'I({lhs_clean}) ~ {" + ".join(rhs_names)} -1'

