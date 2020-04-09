from collections import defaultdict

import statsmodels.formula.api as smf
import pandas
from itertools import combinations, product
import magma
import re
from fault import RealKind
from ast import literal_eval

class Regression():
    # statsmodels gets confused if you try to use '1' to mean a column of 
    # ones, so we manually add a column of ones with the following name
    one_literal = 'const_1'

    # @classmethod
    def parse_parameter_algebra(self, lhs, rhs):
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

    # @classmethod
    def get_spice_name(cls, port):
        # TODO: this has to match the way fault does it in spice_target
        # I think that this conversion should be broken out into a method in fault
        if isinstance(port.name, magma.ref.ArrayRef):
            bus_name_full = str(port.name.array.name)
            bus_name = bus_name_full.split('.')[-1]
            bus_index = port.name.index
            return '%s<%d>' % (bus_name, bus_index)
        else:
            return str(port)

    # @classmethod
    def get_optional_pin_expression(cls, template):
        '''
        Given the magma circuit template, look at the optional pins and create a
        (string) R expression for how the analog pins affect something
        '''

        # TODO pull these out of some dict
        analog_order = 1
        interaction_a_a = False
        interaction_a_ba = False
        interaction_ba_ba = False

        opt_a, opt_ba = template.inputs_analog, template.inputs_ba

        terms = [cls.one_literal]
        for a_port in opt_a:
            a = cls.get_spice_name(a_port)
            for i in range(1, analog_order + 1):
                if i == 1:
                    terms.append(a)
                else:
                    terms.append('I(%s**%d)' % (a, i))
                    #terms.append('%s**%d' % (a, i))

        for ba_port in opt_ba:
            ba = cls.get_spice_name(ba_port)
            terms.append(ba)


        def interact(iterator):
            for a_port, b_port in iterator:
                a, b = cls.get_spice_name(a_port), cls.get_spice_name(b_port)
                terms.append('%s:%s' % (a, b))

        if interaction_a_a:
            interact(combinations(opt_a, 2))
        if interaction_ba_ba:
            interact(combinations(opt_ba, 2))
        if interaction_a_ba:
            interact(product(opt_a, opt_ba))

        return ' + '.join(terms)


    # @classmethod
    def clean_string(self, s):
        # discrepency between the way spice and magma do braces
        # also Patsy won't use < in a variable name
        temp = s.replace('<', '_').replace('>', '_')
        return temp.replace('[', '_').replace(']', '_')


    def convert_required_ba(self, template, rhs):
        '''
        when the parameter algebra contains an Array (most likely ba required input),
        we have to break that term into multiple terms.
        This function edits rhs in place to split terms with an array.
        '''
        to_be_deleted = set()
        to_be_added = {}
        for arr_req in template.inputs_analog + template.inputs_ba:
            if isinstance(arr_req.name, magma.ref.ArrayRef):
                bus_name = str(arr_req.name.array.name)
                inst_name = str(arr_req.name).split('.')[-1]
                new_name = self.clean_string(inst_name)
                search_str = r'\b' + bus_name + r'\b'

                for key_term in rhs:
                    if re.search(search_str, key_term):
                        to_be_deleted.add(key_term)
                        new_key_term = re.sub(search_str, new_name, key_term)
                        to_be_added[new_key_term] = rhs[key_term] + self.component_tag + inst_name
        for d in to_be_deleted:
            del rhs[d]
        for k, v in to_be_added.items():
            rhs[k] = v

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

    def __init__(self, template, test, data):
        '''
        Incoming data should be of the form 
        {pin:[x1, ...], ...}
        '''

        def get_name(x):
            if type(x) == str:
                return x
            else:
                return template.get_name_template(x)
        self.component_tag = '_component_'
        data = {get_name(k): v for k, v in data.items()}
        data[self.one_literal] = [1 for _ in list(data.values())[0]]
        data = {self.clean_string(k):v for k,v in data.items()}
        self.df = pandas.DataFrame(data)

        self.consts ={}
        def create_const(rhs):
            '''
            pandas gets confused when you put a constant in the formula,
            so for each constant we create a dedicated column in the df.
            :param rhs: dictionary of expressions which may contain consts
            :return: nothing; it modifies rhs and self.df in place
            '''
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
        for lhs, rhs in test.parameter_algebra.items():
            lhs, rhs = self.parse_parameter_algebra(lhs, rhs)

            optional_pin_expr = self.get_optional_pin_expression(template)

            self.convert_required_ba(template, rhs)
            create_const(rhs)
            print('param algebra is now', lhs, rhs)
            print(self.df)

            formula = self.make_formula(lhs, rhs, optional_pin_expr)
            #print('formula was', formula)
            #formula = 'amp_output ~ adj + constant_ones'
            #print('changed to ', formula)

            stats_model = smf.ols(formula, self.df)
            stat_results = stats_model.fit()
            #print(results.summary())
            result = self.parse_coefs(stat_results, rhs)
            for k,v in result.items():
                assert not k in results, 'Parameter %s found in multiple parameter algebra formulas'
                results[k] = v

        self.condense_required_ba(results)

        print('param\tterm\tcoef')
        for param,d in results.items():
            for partial_term_optional, coef in d.items():
                print('%s\t%s\t%.3f' % (param, partial_term_optional, coef))

        # TODO dump res to a yaml file
        self.results = results

    # @classmethod
    def un_create_const(self, name):
        if name in self.consts:
            return self.consts[name]
        else:
            return name

    # @classmethod
    def make_formula(self, lhs, rhs, optional_pin_expr):
        terms = []
        for expr, param in rhs.items():
            terms.append('%s:(%s)'%(expr, optional_pin_expr))

        # the constant term should already be included as constant_ones somewhere
        formula = '%s ~ %s -1' % (lhs, ' + '.join(terms))
        return self.clean_string(formula)
        
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


        

