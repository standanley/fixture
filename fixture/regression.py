import statsmodels.formula.api as smf
import pandas
from itertools import combinations, product
import magma
import re


class Regression():
    # statsmodels gets confused if you try to use '1' to mean a column of 
    # ones, so we manually add a column of ones with the following name
    one_literal = 'constant_ones'

    @classmethod
    def parse_parameter_algebra(self, f):
        ''' Parse a formula to find coefficients of params.
        I didn't want to write my own parser but we do need to understand
        parentheses so regex isn't good enough, and patsy won't parse it
        unless it knows all the variable names ahead of time.
        Ex:
        input : 'out_single ~ gain:in_single + offset'
        output: ('out_single', {'in_single': 'gain', '1': 'offset'})
        '''

        depth = 0
        prev = 0
        STATE = 'lhs'
        rhs = {}

        def get_param():
            param = f[prev+1:i].strip()
            # TODO: check for non \w chars
            return param

        for i, char in enumerate(f + '+'):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            if depth != 0:
                continue

            if STATE == 'lhs':
                if char == '~':
                    lhs = f[:i].strip()
                    prev = i
                    STATE = 'param'
                    #print('going to param after', lhs)
            elif STATE == 'param':
                if char == ':':
                    param = get_param()
                    prev = i
                    STATE = 'expr'
                    #print('going to expr after', param)
                elif char == '+':
                    param = get_param()
                    rhs[self.one_literal] = param
                    prev = i
                    #print('staying param after', param)
            elif STATE == 'expr':
                if char == '+':
                    expr = f[prev+1:i].strip()
                    prev = i
                    rhs[expr] = param
                    STATE = 'param'
                    #print('going to param after ', expr)
        assert depth == 0, 'Unmatched paren in formula "%s"' % f
        assert STATE == 'param', 'Unexpected end while parsing formula "%s"' % f

        return (lhs, rhs)

    @classmethod
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

    @classmethod
    def get_optional_pin_expression(cls, dut):
        '''
        Given the magma circuit dut, look at the optional pins and create a 
        (string) R expression for how the analog pins affect something
        '''

        # TODO pull these out of some dict
        analog_order = 3
        interaction_a_a = False
        interaction_a_ba = False
        interaction_ba_ba = False

        #print(dut.optional_a)
        #print(dut.optional_ba)

        terms = [cls.one_literal]
        for a_port in dut.optional_a:
            a = cls.get_spice_name(a_port)
            for i in range(1, analog_order + 1):
                if i == 1:
                    terms.append(a)
                else:
                    terms.append('I(%s**%d)' % (a, i))

        for ba_port in dut.optional_ba:
            ba = cls.get_spice_name(ba_port)
            terms.append(ba)


        def interact(iterator):
            for a_port, b_port in iterator:
                a, b = cls.get_spice_name(a_port), cls.get_spice_name(b_port)
                terms.append('%s:%s' % (a, b))

        if interaction_a_a:
            interact(combinations(dut.optional_a, 2))
        if interaction_ba_ba:
            interact(combinations(dut.optional_ba, 2))
        if interaction_a_ba:
            interact(product(dut.optional_a, dut.optional_ba))

        return ' + '.join(terms)


    @classmethod
    def clean_string(self, s):
        # discrepency between the way spice and magma do braces
        # also Patsy won't use < in a variable name
        temp = s.replace('<', '_').replace('>', '_')
        return temp.replace('[', '_').replace(']', '_')


    def convert_required_ba(self, dut, rhs):
        # when the parameter algebra contains a term with a ba input,
        # we have to break that term into multiple terms
        to_be_deleted = set()
        to_be_added = {}
        for required_ba in dut.required_ba:
            bus_name = str(required_ba.name.array.name)
            inst_name = str(required_ba.name).split('.')[-1]
            new_name = self.clean_string(inst_name)
            search_str = r'\b' + bus_name + r'\b'

            for key_term in rhs:
                if re.search(search_str, key_term):
                    to_be_deleted.add(key_term)
                    new_key_term = re.sub(search_str, new_name, key_term)
                    to_be_added[new_key_term] = rhs[key_term] + '_' + inst_name
        for d in to_be_deleted:
            del rhs[d]
        for k, v in to_be_added.items():
            rhs[k] = v

    def __init__(self, dut, data):
        '''
        Incoming data should be of the form 
        for [in, out]: {pin:[x1, ...], ...}
        '''


        data = {**data[0], **data[1]}
        print('data is', data)
        data[self.one_literal] = [1 for _ in list(data.values())[0]]
        data = {self.clean_string(k):v for k,v in data.items()}
        self.df = pandas.DataFrame(data)
        #print(self.df)

        results = {}
        for params_algebra in dut.parameter_algebra:
            lhs, rhs = self.parse_parameter_algebra(params_algebra)
            optional_pin_expr = self.get_optional_pin_expression(dut)

            print('CALLING CONVERSION', rhs)
            self.convert_required_ba(dut, rhs)

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

        print('param\tterm\tcoef')
        for param,d in results.items():
            for partial_term_optional, coef in d.items():
                print('%s\t%s\t%.3f' % (param, partial_term_optional, coef))

        # TODO dump res to a yaml file

        
    @classmethod
    def make_formula(self, lhs, rhs, optional_pin_expr):
        terms = []
        for expr, param in rhs.items():
            terms.append('%s:(%s)'%(expr, optional_pin_expr))

        # the constant term should already be included as constant_ones somewhere
        formula = '%s ~ %s -1' % (lhs, ' + '.join(terms))
        return self.clean_string(formula)
        
    @classmethod
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
        #print('\nhere are the coefficient names')
        #for t,c in coefs.items():
        #    print('\t'+t)
        for term, coef in coefs.items():
            param, partial_term_optional = get_relevant_param(term)
            #print('%s\t%s\t%.3f' % (param, partial_term_optional, coef))
            res[param][partial_term_optional] = coef
        return res


        

