from collections import defaultdict

import statsmodels.formula.api as smf
import pandas
from itertools import combinations, product
import magma
import re
from ast import literal_eval

from fixture.signals import SignalArray


class Regression:
    # statsmodels gets confused if you try to use '1' to mean a column of 
    # ones, so we manually add a column of ones with the following name
    one_literal = 'const_1'

    @staticmethod
    def parse_parameter_algebra(lhs, rhs):
        '''
        Removes spaces, wraps in I(), and flips RHS terms so expr is the key
        :param f:
        :return:
        '''
        def clean(s):
            return f'I({s.replace("" "", "")})'

        if rhs == 'const':
            rhs = {lhs:'1'}

        res = {}
        for k,v in rhs.items():
            res[clean(v)] = k.replace(' ', '_')
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
        opt_a = [cls.regression_name(s) for s in opt_signals_flat if s.type_ == 'analog']
        opt_ba = [cls.regression_name(s) for s in opt_signals_flat if s.type_ == 'binary_analog']

        terms = [cls.one_literal]
        for a in opt_a:
            for i in range(1, analog_order + 1):
                if i == 1:
                    terms.append(a)
                else:
                    terms.append('I(%s ** %d)' % (a, i))

        for ba in opt_ba:
            terms.append(ba)


        def interact(iterator):
            for a, b in iterator:
                terms.append('%s:%s' % (a, b))

        if interaction_a_a:
            interact(combinations(opt_a, 2))
        if interaction_ba_ba:
            interact(combinations(opt_ba, 2))
        if interaction_a_ba:
            interact(product(opt_a, opt_ba))

        return ' + '.join(terms)

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

    def convert_required_ba(self, test, rhs):
        '''
        when the parameter algebra contains an Array (most likely ba required input),
        we have to break that term into multiple terms.
        This function edits rhs in place to split terms with an array.
        The param is also split accordingly, and tagged with self.component_tag.
        '''

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
                        bit_name = self._clean_string(bit.template_name)
                        new_key_term = re.sub(search_str, bit_name, key_term)
                        new_param = rhs[key_term] + self.component_tag + bit_name
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
            return s
        if s.template_name is not None:
            return s.template_name
        else:
            assert s.spice_name is not None, f'Signal {s} has neither template nor spice name!'
            return s.spice_name

    def __init__(self, template, test, data):
        '''
        Incoming data should be of the form 
        {pin:[x1, ...], ...}
        '''

        self.component_tag = '_component_'
        self.name_mapping = {}
        # translate from circuit names to template names
        data = {self.regression_name(k): v for k, v in data.items()}
        data[self.one_literal] = [1 for _ in list(data.values())[0]]
        data = {self._clean_string(k):v for k,v in data.items()}
        self.df = pandas.DataFrame(data)

        # look for optional outputs
        for s in test.signals.auto_measure():
            if s.spice_name in self.df.columns:
                # add parameter_algebra entry
                test.parameter_algebra[s.spice_name] = {f'{s.spice_name}_meas': '1'}

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

        results = {}
        results_models = {}
        regression_dicts = []
        for lhs, rhs in test.parameter_algebra.items():
            lhs_not_wrapped = lhs
            lhs, rhs = self.parse_parameter_algebra(lhs, rhs)

            # TODO I want a better model for this so we can evaluate later
            # e.g. given the inputs, what does the pin_expr evaluate to?
            optional_pin_expr = self.get_optional_pin_expression(template)

            self.convert_required_ba(test, rhs)
            create_const(rhs)

            formula = self.make_formula(lhs, rhs, optional_pin_expr)

            #df_row_mask = self.df[lhs_not_wrapped] != None

            df_row_mask = ~ self.df[lhs_not_wrapped].isnull()
            df_filtered = self.df[df_row_mask]

            stats_model = smf.ols(formula, df_filtered)
            #stats_model_unfiltered = smf.ols(formula, self.df)
            stat_results = stats_model.fit()
            result = self.parse_coefs(stat_results, rhs)
            for k, v in result.items():
                assert not k in results, f'Parameter {k} found in multiple parameter algebra formulas'
                results[k] = v

            results_models[lhs] = stat_results

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

        self.condense_required_ba(results)

        # change regression names back to spice names
        for param in list(results.keys()):
            terms = results[param]
            terms_new = {self.revert_names(term): coef
                         for term, coef in terms.items()}
            results[param] = terms_new

        data2 = data
        for regression_dict in regression_dicts:
            for term, values in regression_dict.items():
                if term in data2:
                    # TODO issues here when the values have been filtered
                    # differently by different parameter algebra entries
                    # Not sure what the thing to do is
                    #assert all(data2[term] == values)
                    pass
                else:
                    data2[term] = values

        # for cases where we have a term like (A*adj+B)*in**2
        # we should make sure that regression_data2 contains in**2
        # I think right now it will always be in the columns as in**2*const_1
        # because of that B with no other coef, but even if that changes in
        # the future we should still include in**2 in our columns
        for column in list(data2.keys()):
            if column[-1*len(self.one_literal)-1:] == '*'+self.one_literal:
                data2[column[:-1*len(self.one_literal)-1]] = data2[column]

        # remove spaces from column names
        data2 = {k.replace(' ', ''): v for k, v in data2.items()}



        # TODO dump res to a yaml file
        self.regression_dataframe = pandas.DataFrame(data2)
        self.results = results
        self.results_models = results_models


    # @classmethod
    def make_formula(self, lhs, rhs, optional_pin_expr):
        terms = []
        for expr, param in rhs.items():
            terms.append('%s:(%s)'%(expr, optional_pin_expr))

        # the constant term should already be included as constant_ones somewhere
        formula = '%s ~ %s -1' % (lhs, ' + '.join(terms))
        return self._clean_string(formula)
        
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


        

