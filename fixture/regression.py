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
    component_tag = '_component_'

    @staticmethod
    def parse_parameter_algebra(lhs, rhs):
        '''
        Removes spaces, wraps in I(), and flips RHS terms so expr is the key
        Also replaces braces with underscore
        :param f:
        :return:
        '''
        def clean(s):
            s2 = Regression.clean_string(s)
            return f'I({s2.replace("" "", "")})'

        if rhs == 'const':
            rhs = {lhs:'1'}

        res = {}
        for k,v in rhs.items():
            res[clean(v)] = clean(k).replace(' ', '_')
        return (clean(lhs), res)

    #@staticmethod
    #def get_spice_name(port):
    #    # TODO: this has to match the way fault does it in spice_target
    #    # I think that this conversion should be broken out into a method in fault
    #    if isinstance(port.name, magma.ref.ArrayRef):
    #        bus_name_full = str(port.name.array.name)
    #        bus_name = bus_name_full.split('.')[-1]
    #        bus_index = port.name.index
    #        return '%s<%d>' % (bus_name, bus_index)
    #    else:
    #        if type(port) == magma.DigitalMeta:
    #            return port.name
    #        print('returning', str(port))
    #        return str(port)

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
        #opt_a = [cls.regression_name(s) for s in opt_signals_flat if s.type_ == 'analog']
        #opt_ba = [cls.regression_name(s) for s in opt_signals_flat if s.type_ == 'binary_analog']
        opt_a = [s for s in opt_signals_flat if s.type_ == 'analog']
        opt_ba = [s for s in opt_signals_flat if s.type_ == 'binary_analog']

        terms = [cls.one_literal]
        for a in opt_a:
            for i in range(1, analog_order + 1):
                if i == 1:
                    terms.append(a)
                else:
                    #terms.append('I(%s ** %d)' % (a, i))
                    terms.append(tuple([a]*i))

        for ba in opt_ba:
            terms.append(ba)


        def interact(iterator):
            for a, b in iterator:
                #terms.append('%s:%s' % (a, b))
                terms.append((a, b))

        if interaction_a_a:
            interact(combinations(opt_a, 2))
        if interaction_ba_ba:
            interact(combinations(opt_ba, 2))
        if interaction_a_ba:
            interact(product(opt_a, opt_ba))

        #return ' + '.join(terms)
        return terms

    @staticmethod
    def clean_string(s):
        # discrepency between the way spice and magma do braces
        # also Patsy won't use < in a variable name
        temp = s.replace('<', '_').replace('>', '_')
        final = temp.replace('[', '_').replace(']', '_')
        return final

    def _clean_string(self, s):
        cleaned = Regression.clean_string(s)
        self.name_mapping[cleaned] = s
        return cleaned

    def revert_names(self, text):
        # look for cleaned names in text and replace them with originals
        # sometimes the text is an expression incuding the names, so it's not
        # good enough to look for the text in the name_mapping keys
        # This is very brute-force
        for new, old in self.name_mapping.items():
            text = text.replace(new, old)
        return text

    def clean_exog_name(self, name):
        def strip_I(s):
            if s[:2] == 'I(' and s[-1] == ')':
                return s[2:-1]
            else:
                return s
        tokens = [self.revert_names(strip_I(t)) for t in name.split(':')]
        return '*'.join(tokens)

    @classmethod
    def convert_required_ba(cls, test, rhs):
        '''
        when the parameter algebra contains an Array (most likely ba required input),
        we have to break that term into multiple terms.
        This function edits rhs in place to split terms with an array.
        The param is also split accordingly, and tagged with cls.component_tag.
        '''

        # TODO not sure classmethod is okay here because of self.name_mapping
        template_bus_names = [n for n in test.template.required_ports if
            isinstance(test.signals.from_template_name(n), SignalArray)]

        to_be_deleted = set()
        to_be_added = {}
        for key_term in rhs:
            for n in template_bus_names:
                search_str = '\\b' + n + '\\b'
                if re.search(search_str, key_term):
                    # the required bus is in this term
                    to_be_deleted.add(key_term)
                    for bit in test.signals.from_template_name(n):
                        bit_name = bit.template_name #cls._clean_string(bit.template_name)
                        new_key_term = re.sub(search_str, bit_name, key_term)
                        new_param = rhs[key_term] + cls.component_tag + bit_name
                        to_be_added[new_key_term] = new_param

        for key in to_be_deleted:
            del rhs[key]
        rhs.update(to_be_added)

    def condense_required_ba(self, results):
        buss = defaultdict(list)
        for k, v in results.items():
            if self.component_tag in k:
                # k is one bit of required ba
                param, bit = k.split(self.component_tag)
                buss[param].append((k, bit))
        for bus, bits in buss.items():
            assert bus not in results, f'Cannot have entire bus {bus} and bus components both in results'
            results[bus] = {}
            for name, bit in bits:
                print(name, bit, results[name])
                terms = results.pop(name)
                for coef, value in terms.items():
                    results[bus][f'{bit}*{coef}'] = value

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
        return f'{name}_vec[{vec_i}]'

    @staticmethod
    def vector_parameter_name_input(name, vec_i, signal):
        return f'{name}_vec[{vec_i}]'

    def get_terms(self, rhs, opt):
        # essentially an outer product of rhs and opt
        # we delete ones so a const one is just an empty tuple
        # For each term we need
        # 1) unique name (can assume products are unique)
        # 2) tuple of product components (for calculating value)
        # 3) Which param it is part of
        # 4) The optional pin it corresponds to (for reporting to user)
        ones = ['1', self.one_literal]
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
        ## translate from circuit names to template names
        #data = {self.regression_name(k): v for k, v in data.items()}
        #data[self.one_literal] = 1
        #data = {self._clean_string(k):v for k,v in data.items()}
        #self.df = pandas.DataFrame(data)

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

        def create_const(rhs):
            '''
            pandas gets confused when you put a constant in the formula,
            so for each constant we create a dedicated column in the df.
            :param rhs: dictionary of expressions which may contain consts
            :return: nothing; it modifies rhs and self.df in place
            '''
            # TODO this edits a dict as it's being traversed
            for expr in rhs:
                expr_literal = expr[2:-1]
                try:
                    const = literal_eval(expr_literal)
                    column_name = f'const_{expr_literal}'
                    self.df[column_name] = const
                    param = rhs[expr]
                    del rhs[expr]
                    rhs[column_name] = param
                    #print(const)
                    self.consts[column_name] = expr_literal
                except ValueError:
                    # not a constant
                    pass


        # results[lhs][param][opt] = coef
        # example:
        # {'amp_output':
        #     {'dcgain':
        #          {'const_1': 0.2658061976765506,
        #           <vdd>: -0.011425713964578879},
        #      }
        # }

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
            regression_data = pandas.DataFrame(regression_data_dict)

            #lhs, rhs = self.parse_parameter_algebra(lhs, rhs)

            # TODO I want a better model for this so we can evaluate later
            # e.g. given the inputs, what does the pin_expr evaluate to?

            #self.convert_required_ba(test, rhs)
            #create_const(rhs)

            formula = self.make_formula(lhs_clean, [x[0] for x in rhs_info])

            #df_row_mask = self.df[lhs_not_wrapped] != None

            df_row_mask = ~ regression_data[lhs_clean].isnull()
            df_filtered = regression_data[df_row_mask]

            stats_model = smf.ols(formula, df_filtered)
            #stats_model_unfiltered = smf.ols(formula, self.df)
            stat_results = stats_model.fit()
            #result = self.parse_coefs(stat_results, rhs)
            results_entry = results[lhs]
            for name, coef in stat_results.params.items():
                term, param, opt = self.info_mapping[name]
                assert opt not in results_entry[param], f'Parameter {lhs}->{param}->{opt} found in multiple parameter algebra formulas'
                results_entry[param][opt] = coef

            results_models[lhs] = stat_results

            # Now, we would like to combine all regression entries into one big
            # dict. BUT we sometimes have issues with nans ...
            regression_dicts.append(regression_data)
            continue


            def fill_out(values):
                # insert NaNs according to df_row_mask
                i = 0
                new_values = []
                for m in df_row_mask:
                    if m:
                        new_values.append(values[i])
                        i += 1
                    else:
                        new_values.append(float('nan'))
                return new_values
            regression_dict = {self.clean_exog_name(name): fill_out(values) for
                                    name, values in zip(stats_model.exog_names,
                                                        stats_model.exog.T)
                               }
            regression_dicts.append(regression_dict)

        #self.condense_required_ba(results)

        ## change regression names back to spice names
        #for param in list(results.keys()):
        #    terms = results[param]
        #    terms_new = {self.revert_names(term): coef
        #                 for term, coef in terms.items()}
        #    results[param] = terms_new

        #data2 = data
        #for regression_dict in regression_dicts:
        #    for term, values in regression_dict.items():
        #        if term in data2:
        #            # TODO issues here when the values have been filtered
        #            # differently by different parameter algebra entries
        #            # Not sure what the thing to do is
        #            #assert all(data2[term] == values)
        #            pass
        #        else:
        #            data2[term] = values

        ## for cases where we have a term like (A*adj+B)*in**2
        ## we should make sure that regression_data2 contains in**2
        ## I think right now it will always be in the columns as in**2*const_1
        ## because of that B with no other coef, but even if that changes in
        ## the future we should still include in**2 in our columns
        #for column in list(data2.keys()):
        #    if column[-1*len(self.one_literal)-1:] == '*'+self.one_literal:
        #        data2[column[:-1*len(self.one_literal)-1]] = data2[column]

        ## remove spaces from column names
        #data2 = {k.replace(' ', ''): v for k, v in data2.items()}

        # combine the individual dataframes into a big one
        # when they have hte same column heading, assert that the data is equal
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


    # @classmethod
    def make_formula(self, lhs_clean, rhs_names):
        return f'I({lhs_clean}) ~ {" + ".join(rhs_names)} -1'

        # terms = []
        # for expr, param in rhs.items():
        #     terms.append('%s:(%s)'%(expr, optional_pin_expr))

        # # the constant term should already be included as constant_ones somewhere
        # formula = '%s ~ %s -1' % (lhs, ' + '.join(terms))
        # return self._clean_string(formula)
        
    # @classmethod
    def parse_coefs(self, results, rhs):
        #print('parsing coefs with rhs', rhs)
        def get_relevant_param(term):
            ''' Break the term into the param half and the optional pin half,
            then return the corresponding param name and optional pin half 
            '''
            def norm(s):
                ''' put string in normal form, i.e. no whitespace '''
                return s.replace(' ', '')

            term = norm(term)
            for partial_term_param, param in rhs.items():
                partial_term_param = norm(partial_term_param)
                if term.startswith(partial_term_param):
                    partial_term_optional = term[len(partial_term_param)+1:]
                    if partial_term_optional == '':
                        partial_term_optional = self.one_literal
                    return (param, partial_term_optional)
            assert False, 'Error: Could not find a param for term %s' % term

        res = {param:{} for param in rhs.values()}

        coefs = results.params
        for term, coef in coefs.items():
            param, partial_term_optional = get_relevant_param(term)
            if partial_term_optional in self.consts:
                partial_term_optional = self.consts[partial_term_optional]
            res[param][partial_term_optional] = coef
        return res


        

