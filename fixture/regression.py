import itertools
from collections import defaultdict
from functools import reduce

import pandas.core.computation.ops
import statsmodels.formula.api as smf
import pandas
from itertools import combinations, product
import magma
import re
from ast import literal_eval

from fixture.signals import SignalArray, SignalIn, CenteredSignalIn
import operator

from fixture.optional_fit import get_optional_expression_from_signals, LinearExpression, HeirarchicalExpression


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
        #opt_ba = [s for s in opt_signals_flat if s.type_ == 'binary_analog']
        opt_ba_notflat = [s for s in opt_signals if s.type_ == 'binary_analog']
        def flatten_ba(s):
            if s.info['bus_type'] == 'signed_magnitude':
                flat_ver = list(s.flat)
                sign = flat_ver.pop()
                return flat_ver + [sign] + [(sign, x) for x in flat_ver]
            elif s.info['bus_type'] == 'binary':
                if 'break_binary_for_bits' in s.info:
                    ans = []
                    expanded_i = s.info['break_binary_for_bits']
                    unexpanded = [b for i, b in enumerate(s) if i not in expanded_i]
                    for mask in itertools.product(range(2), repeat=len(expanded_i)):
                        prod = [s[expanded_i[i]] for i, include in enumerate(mask) if include==1]
                        # don't include constant, that is done outside of flatten_ba
                        if prod != []:
                            ans.append(tuple(prod))
                        for b in unexpanded:
                            ans.append((b, *prod))
                    return ans
                else:
                    return list(s.flat)
            elif s.info['bus_type'] == 'binary_exact':
                # convert bits to a number and use that number instead
                return [s]
            else:
                return list(s.flat)
        opt_ba = [term for s in opt_ba_notflat for term in flatten_ba(s)]

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


    @classmethod
    def get_optional_pin_expression2(cls, template, param_name):
        '''
        Given the magma circuit template, look at the optional pins and create a
        (string) R expression for how the analog pins affect something
        '''

        # TODO option for interaction terms, etc.

        assert False, 'This logic was moved to config_parse.parse_optional_config_info'
        opt_signals = template.signals.optional_expr()
        opt_signals_flat = [s for x in opt_signals for s in (x if isinstance(x, SignalArray) else [x])]

        expr = get_optional_expression_from_signals(opt_signals, param_name)
        return expr


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
        #return f'{name}_outvec[{vec_i}]'
        return f'{name}_{signal.friendly_name()}'

    @staticmethod
    def vector_parameter_name_input(name, vec_i, signal):
        #return f'{name}_invec[{vec_i}]'
        return f'{name}_{signal.friendly_name()}'

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
        pa = {lhs: rhs.copy() for lhs, rhs in test.parameter_algebra_vectored.items()}
        # now handled in template_master
        #for s in test.signals.auto_measure():
        #    if s.spice_name in data.columns:
        #        # add parameter_algebra entry
        #        pa[s.spice_name] = {f'{s.spice_name}_meas': '1'}
        #    else:
        #        assert False, f'Where to get value for auto_measure {s}?'


        #def get_center(s):
        #    # for now, just do real types with range
        #    if (isinstance(s, SignalIn)
        #        and isinstance(s.value, tuple)
        #        and len(s.value) == 2):
        #        nom = sum(s.value) / 2
        #        return nom
        #    else:
        #        return 0
        def get_term_centering(term):
            # This was an attempt to keep track of the effects of centering on regression results
            # I decided that it's too complicated to track the effects of these shifts, and
            # we should just do regression twice instead
            # One issue is that it's possible (unlikely?) for the shifts to introduce new terms,
            # and in those cases the two runs of results will have different predictions
            # -----
            # for the term (cm_in,) the centering is just cm_in_NOM
            # for the term (diff_in, cm_in, cm_in) it's:
            # 2*diff_in*cm_in * cm_in_NOM  +  diff_in * cm_in_NOM^2
            # So we need to call out which terms need to be adjusted, and coefficients:
            # {(): cm_in_NOM}
            # {(diff_in, cm_in): 2*cm_in_NOM, (diff_in,): cm_in_NOM^2}
            pass

        #to_center = {}
        #for lhs, rhs in test.parameter_algebra_vectored.items():
        #    for param, term:


        center_mapping = {}
        for lhs, rhs in test.parameter_algebra_final.items():
            for s in rhs.input_signals:
                if isinstance(s, CenteredSignalIn):
                    assert False, 'todo'

        #for lhs, rhs in pa.items():
            #for param, term in rhs.items():
            #    for factor in term:
            #        if isinstance(factor, CenteredSignalIn):
            #            # create a column for this centered thing
            #            if factor.ref in center_mapping:
            #                continue
            #            orig_column = data[factor.ref]
            #            nom = sum(factor.ref.value) / 2
            #            new_column = orig_column - nom
            #            #new_name = self.regression_name(factor)
            #            #data[factor] = new_column
            #            data.insert(len(data.columns), factor, new_column)
            #            center_mapping[factor.ref] = factor


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
        results_expr = {}
        #results = {lhs: defaultdict(dict) for lhs in pa}
        #results_models = {}
        #regression_dicts = []
        #for lhs, rhs in pa.items():
        #    lhs_clean = self.clean_string(lhs)
        #    regression_data_dict = {lhs_clean: data[lhs]}
        #    self.info_mapping = {}

        #    optional_pin_expr = self.get_optional_pin_expression(template)
        #    rhs_info = self.get_terms(rhs, optional_pin_expr)
        #    for name, term, param, opt in rhs_info:
        #        assert name not in self.info_mapping, f'Duplicate term: {name}'
        #        self.info_mapping[name] = (term, param, opt)
        #        try:
        #            column = reduce(operator.mul,
        #                            [data[x] for x in term],
        #                            data[self.one_literal])
        #        except KeyError as e:
        #            missing = str(e.args[0])
        #            keys = [str(k) for k in data]
        #            raise KeyError(f'Could not find {missing} in data, available keys are {keys}')
        #        regression_data_dict[name] = column

        #        # also put term elements into the dict, useful for plotting
        #        if len(term) > 1:
        #            for term_element in term:
        #                element_name = self.regression_name(term_element)
        #                regression_data_dict[element_name] = data[term_element]

        #        # also put centered elements into the dict; useful for plotting
        #        for orig in center_mapping:
        #            regression_data_dict[self.regression_name(orig)] = data[orig]

        #    # TODO if every instance of 'vdd' is in a product term, should we
        #    # still add 'vdd' to the dataframe by itself?
        #    regression_data = pandas.DataFrame(regression_data_dict)


        #    # TODO I stole the filtering code from this spot

        #    ## build up parameter algebra expression
        #    #optional_exprs = []
        #    #param_names = []
        #    #for thing in rhs:
        #    #    #param = make_optional_expression()
        #    #    param = self.get_optional_pin_expression2(template, thing)
        #    #    optional_exprs.append(param)
        #    #    param_names.append(thing)
        #    print('TODO next for loop must be combined with this; scope of df_filtered is messed up now')

        for lhs, rhs in test.parameter_algebra_final.items():

            #df_row_mask = ~ regression_data[lhs_clean].isnull()
            ## TODO when I blank out entries in the data spreadsheet they appear as nan, but those rows aren't filtered. Is that bad?
            #df_filtered = regression_data[df_row_mask]

            # TODO how to handle rows with nan? I copied this from old code
            lhs_clean = self.clean_string(lhs)
            df_row_mask = ~ data[lhs_clean].isnull()
            # TODO when I blank out entries in the data spreadsheet they appear as nan, but those rows aren't filtered. Is that bad?
            df_filtered = data[df_row_mask]


            # TODO just stole some code from here for config_parse
            total_expr = rhs

            verilog_const_names = [f'c[{i}]' for i in range(total_expr.NUM_COEFFICIENTS)]
            verilog = total_expr.verilog(lhs_clean, verilog_const_names)

            # TODO these next 3 lines were commented out and I'm not sure why
            #  I was using lhs_data in place of expr_fit_data
            #  Update: I think it's because fit does this lookup internally
            #input_names = [self.regression_name(s) for s in total_expr.input_signals]
            #expr_fit_data = [df_filtered[input_name] for input_name in input_names]
            #expr_fit_data = np.array(expr_fit_data)
            lhs_data = df_filtered[lhs_clean]

            # do the regression!
            print('Starting parameter fit')
            coefs_fit = total_expr.fit_by_group(df_filtered, lhs_data)
            #total_expr.coefs_fit = coefs_fit
            for line in verilog:
                print(line)
            for n, v in zip(verilog_const_names, total_expr.x_opt):
                print(f'{n} = {v};')
            print()
            results_expr[lhs] = total_expr

        self.expr_dataframe = data
        self.results_expr = results_expr


    def make_formula(self, lhs_clean, rhs_names):
        return f'I({lhs_clean}) ~ {" + ".join(rhs_names)} -1'

