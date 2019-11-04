import statsmodels.formula.api as smf
import pandas
from itertools import combinations, product
import magma


class Regression():

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
                    rhs['1'] = param
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
        # TODO pull these out of some dict
        analog_order = 3
        interaction_a_a = False
        interaction_a_ba = False
        interaction_ba_ba = False

        terms = []
        for a_port in dut.inputs_ranged:
            a = cls.get_spice_name(a_port)
            for i in range(1, analog_order + 1):
                if i == 1:
                    terms.append(a)
                else:
                    terms.append('I(%s^%d)' % (a, i))

        for ba_port in dut.inputs_ba:
            ba = cls.get_spice_name(ba_port)
            terms.append(ba)


        def interact(iterator):
            for a_port, b_port in iterator:
                a, b = cls.get_spice_name(a_port), cls.get_spice_name(b_port)
                terms.append('%s:%s' % (a, b))

        if interaction_a_a:
            interact(combinations(dut.inputs_ranged, 2))
        if interaction_ba_ba:
            interact(combinations(dut.inputs_ba, 2))
        if interaction_a_ba:
            interact(product(dut.inputs_ranged, dut.inputs_ba))

        return ' + '.join(terms)



    def __init__(self, data, params_algebra):
        '''
        Incoming data should be of the form 
        for [in, out]: {pin:[x1, ...], ...}
        '''
        data = {**data[0], **data[1]}
        self.df = pandas.DataFrame(data)

        lhs, rhs = Regression.parse_parameter_algebra(params_algebra)
        
        
        formula = 'out = in_:I(1) + 1:1'
        smf.ols(formula, data)
        smf.run()
        print(smf.summary())
        

