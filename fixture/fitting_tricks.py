from abc import ABC, abstractmethod
import sympy
from sympy import Symbol, Add, Mul
from sympy.functions.elementary.piecewise import ExprCondPair
import numpy as np


def parse_linear(ast, input_symbols, coef_symbols):
    '''
    Look for an ast that is sum(in_ast*coef_sym), possibly with an offset
    Note that in_ast is any ast that is a function only of input_symbols
    I sometimes call this a "regressable" function
    Return None if the ast is not of this form
    '''
    class LinearAst:
        def __init__(self, input_asts, coefs, offset):
            assert len(input_asts) == len(coefs)
            assert offset is None or isinstance(offset, Symbol)
            self.input_funs = input_asts
            self.coefs = coefs
            self.offset = offset

    def is_input_expr(child):
        for sym in child.free_symbols:
            if sym not in input_symbols:
                return False
        return True

    input_asts = []
    coefs = []
    offsets = []
    if not ast.is_Add:
        return None
    for arg in ast.args:
        if isinstance(arg, Symbol):
            offsets.append(arg)
            continue

        elif arg.is_Mul:
            cs = []
            other = []
            for child in arg.args:
                if child in coef_symbols:
                    cs.append(child)
                else:
                    if not is_input_expr(child):
                        return None
                    other.append(child)

            if len(cs) != 1:
                return None
            assert len(other) > 0, 'Why didnt sympy optimize this product node out?'
            # I think if len(other)==1 sympy will automatically optimize out
            # the product
            input_ast = Mul(*other)
            input_asts.append(input_ast)
            coefs.append(cs[0])

        else:
            # something in the sum is not a Symbol or a Mul
            return None

    if len(offsets) > 1:
        return None
    offset = None if len(offsets) == 0 else offsets[0]

    return LinearAst(input_asts, coefs, offset)

class InitTrick(ABC):
    #@staticmethod
    #@abstractmethod
    #def check_ast(ast, input_symbols, coef_symbols):
    #    '''
    #    Given an ast, check whether you can apply your trick to it.
    #    '''
    #    pass

    @staticmethod
    @abstractmethod
    def guess_init(ast, input_symbols, coef_symbols, input_data, result_data):
        '''
        Given the data to fit, come up with a guess for the coefficients.
        The normal nonlinear optimizer will be run after this, using the guess
        as the initial point.
        '''
        pass



class FitTrick:
    @staticmethod
    @abstractmethod
    def fit(ast, input_symbols, coef_symbols, input_data, result_data):
        pass







class PiecewiseInit(InitTrick):
    #@staticmethod
    #def check_ast(ast, input_symbols, coef_symbols):
    #    if not isinstance(ast, sympy.Piecewise):
    #        return False
    #    return True


    @staticmethod
    def guess_init(ast, input_symbols, coef_symbols, input_data, result_data):
        return None

        if not isinstance(ast, sympy.Piecewise):
            return None

        if len(input_symbols) != 2:
            return None
        indiff, incm = input_symbols
        maximum = max(input_data[indiff]) + max(input_data[incm])
        minimum = min(input_data[indiff]) + min(input_data[incm])
        #guess = [
        #    1,
        #    2/3*(maximum-minimum)+minimum,
        #    1 / 3 * (maximum - minimum) + minimum,
        #    1,
        #    1,
        #    1,
        #    1,
        #    1,
        #    1
        #]
        guess = [
            1, # gain cm
            1, # gain diff
            #-1.5,  # min height
            #1.5,  # max height
                1 / 3 * (maximum - minimum) + minimum,
                2 / 3 * (maximum - minimum) + minimum,
            1,  # offset
        ]

        return guess

        #mins = min(input_data)
        #maxs = max(input_data)
        less_than_indices = []
        greater_than_indices = []
        true_indices = []
        for case in ast.args:
            assert isinstance(case, ExprCondPair)
            expr, cond = case.args
            # TODO true case
            lhs, rhs = cond.args

            # I think the symbol is always the lhs by sympy sorting
            assert not (isinstance(lhs, Add) and isinstance(rhs, Symbol))
            if not isinstance(lhs, Symbol):
                return None
            rhs_parsed = parse_linear(rhs, input_symbols, coef_symbols)
            if rhs_parsed is None:
                return None



        print('hi')



class FitConstant(FitTrick):
    @staticmethod
    def fit(ast, input_symbols, coef_symbols, input_data, result_data):
        # NOTE this could almost fall under FitLinear, but right now there's
        # an issue with parse_linear so it doesn't see this
        if not isinstance(ast, Symbol):
            return None
        if not ast in coef_symbols:
            return None
        avg = sum(result_data) / len(result_data)
        return np.array([avg])

class FitLinear(FitTrick):

    @staticmethod
    def fit(ast, input_symbols, coef_symbols, input_data, result_data):
        linear_model = parse_linear(ast, input_symbols, coef_symbols)
        if linear_model is None:
            return None

        # Minimize ||Ax-b||
        # A is N (number of datapoints) by M (number of coefs)
        N = len(input_data)
        M = len(coef_symbols)
        assert M == len(linear_model.coefs) + (1 if linear_model.offset is not None else 0), 'Internal error in parse_linear?'
        assert len(result_data) == N
        if N < M:
            print(f'Warning: using only {N} datapoints to fit {M} coefficients for {ast}')

        # it's possible for some of linear_model.input_funs to be functions of
        # the inputs, so we use sympy to calculate the column for each one
        A = np.zeros((N, M))
        input_data_array = input_data[input_symbols].values.T
        for coef in linear_model.coefs:
            lm_i = linear_model.coefs.index(coef)
            cs_i = coef_symbols.index(coef)
            input_fun = linear_model.input_funs[lm_i]
            input_lambda = sympy.lambdify(input_symbols, input_fun)
            input_fun_data = input_lambda(*input_data_array)
            A[:, cs_i] = input_fun_data
        if linear_model.offset is not None:
            # column of ones
            cs_i = coef_symbols.index(linear_model.offset)
            A[:, cs_i] = 1

        b = result_data

        x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

        return x





init_tricks = [PiecewiseInit]
fit_tricks = [FitConstant, FitLinear]
