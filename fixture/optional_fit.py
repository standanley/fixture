import re
from abc import ABC, abstractmethod
from collections import defaultdict, Counter

import sympy
from scipy.optimize import minimize, basinhopping
from scipy.stats import linregress
import numpy as np
import pandas

from sympy import Symbol, lambdify, Mul, Add, diff, simplify

import fixture.regression
from fixture.fitting_tricks import init_tricks, fit_tricks
from fixture.sampler import SampleManager
from fixture.signals import SignalArray, SignalIn, CenteredSignalIn, Signal
from fixture.plot_helper import plt, PlotHelper

PLOT = True

class Expression(ABC):
    # TODO I would like for these to be abstract properties, but also they are
    #  not required until the end of __init__, so I don't think abc can do that
    input_signals = None
    NUM_COEFFICIENTS = None

    x_init = None
    x_opt = None
    last_coef_is_offset = False

    def __init__(self, name):
        # TODO when I override this, instead of calling super(), I've just been
        # rewriting this one line to set the name. Is that bad practice? probably
        self.name = name


    @abstractmethod
    def predict(self, opt_values, coefs):
        # given the values of the optional inputs and the fit coefficients,
        # calculate the linear influence of the optional inputs
        pass

    def predict_from_dict(self, opt_dict, coefs):
        # Similar to predict, but takes the optional inputs as a dictionary or
        # dataframe. This one is more convenient, but the minimizer can get
        # better performance by using predict directly
        opt_values = [opt_dict[s] for s in self.input_signals]
        opt_values = np.array(opt_values)
        # reshape only does something if len(self.input_signals)==0
        opt_values = opt_values.reshape((-1, opt_dict.shape[0]))
        if hasattr(self, 'predict_many'):
            ans = self.predict_many(opt_dict[self.input_signals], coefs)
        else:
            ans = []
            for value_vector in opt_values.T:
                ans.append(self.predict(value_vector, coefs))
        return ans

    def error(self, my_input_data, coefs, result_data):
        # predict_vec = np.vectorize(self.predict, signature='(n),(m)->()')
        # predictions = predict_vec(my_data, coefs)
        predictions = self.predict_many(my_input_data, coefs)
        assert len(result_data) == len(predictions)
        errors = result_data - predictions
        e = sum(errors ** 2)
        return (e / len(result_data)) ** .5

    def fit(self, optional_data, result_data, x_init=None):
        # return a best-fit of the coefficients, i.e. minimize
        # predict(optional_data, coefficients) - result_data
        # return a tuple of (coefficients, offset)
        my_data = optional_data[self.input_signals]
        assert my_data.shape == (len(result_data), len(self.input_signals))
        def error(coefs):
            ans = self.error(my_data, coefs, result_data)
            return ans

        # check fit_tricks and init_tricks
        if not isinstance(self, ConstExpression):
            io_syms = list(self.io_symbols.values())
            coef_syms = self.coefs
            #io_reverse = {sym: sig for sig, sym in io_syms.items()}
            data_by_input_sym = {self.io_symbols[sig]: my_data[sig] for sig in self.input_signals}
            data_by_input_sym = pandas.DataFrame(data_by_input_sym)

            for fit_trick in fit_tricks:
                result = fit_trick.fit(self.ast, io_syms, coef_syms,
                                       data_by_input_sym, result_data)
                if result is None:
                    continue

                # we were able to use this trick! we are done
                #print(f'Successfully fit {self} using {fit_trick}')
                self.x_opt = result
                return result


            for init_trick in init_tricks:
                suggested_init = init_trick.guess_init(self.ast, io_syms, coef_syms,
                                                       data_by_input_sym, result_data)
                if suggested_init is not None:
                    #assert self.x_init is None, 'TODO double suggestion'
                    self.x_init = suggested_init

        if x_init is not None:
            x0 = x_init
        else:
            x0 = self.x_init
        if x0 is None:
            x0 = np.ones(self.NUM_COEFFICIENTS)
        x0 = np.array(x0)

        #import cProfile
        #def to_profile():
        #    return minimize(error, x0)
        #cProfile.runctx('to_profile()', None, locals())
        import time
        start = time.time()
        #print('Minimizing', self.name)
        assert x0 is not None
        #result = minimize(error, x0, method='Nelder-Mead', options={
        #    #'ftol': 1e-8, # Powell
        #    'fatol': 0, # Nelder-Mead
        #    'maxfev': 10**4,
        #    'return_all': True
        #})

        # read about basin hopping temperature to maybe understand this ... I
        # have no idea what I'm doing, and I made up this metric
        # I'm also aiming for a relatively high temperature because I think our
        # minimizer is fairly expensive (???)
        avg_result = sum(result_data)/len(result_data)
        error_of_avg = (sum((result_data - avg_result)**2)/len(result_data))**0.5
        temperature = error_of_avg/2

        # By default, the step size adjustment moves by 10% changes and can only
        # change once every 50 steps. I only expect to do ~10 steps total
        #displace = _basinhopping.RandomDisplacement(stepsize=1.0)
        #take_step_wrapped = _basinhopping.AdaptiveStepsize(displace,
        #                                    interval=3,
        #                                    factor=0.1,
        #                                    verbose=False)
        def random_step(x):
            # modifies x in place and returns it
            # most a coef can change in 1 step is a factor of "factor"
            factor = 3
            for i in range(len(x)):
                if abs(x[i]) < 1e-16:
                    x[i] += np.random.rand() - 0.5
                elif np.random.rand() < 0.20:
                    x[i] = -x[i]
                else:
                    x[i] *= factor ** (np.random.rand()*2 - 1)

            return x


        bounds = []
        for coef in self.coefs:
            if coef in self.bounds_dict:
                bounds.append(self.bounds_dict[coef])
            else:
                bounds.append((None, None))

        #print(f'Error before:', error(x0))
        result = basinhopping(
            error,
            x0,
            niter=10**1,
            T=temperature,
            #take_step=take_step_wrapped,
            take_step=random_step,
            minimizer_kwargs={
                'method': 'Nelder-Mead',
                'bounds': bounds,
                'options': {
                    #'ftol': 1e-8, # Powell
                    'fatol': 0, # Nelder-Mead
                    'maxfev': 10**2,
                    'return_all': False,
                }
            }
        )
        x_opt = result.x

        #print('minimization took', time.time() - start, 'error', error(x_opt))
        self.x_opt = x_opt
        #print(x0)
        #print(x_opt)
        #print()


        if False:
            from fixture.plot_helper import plt

            # TODO this is a temporary plot
            predict_vec = np.vectorize(self.predict, signature='(n),(m)->()')
            predictions = predict_vec(my_data, x0)
            x_label = my_data.columns[0]
            plt.title('TEMP looking at x_init')
            plt.scatter(my_data[x_label], result_data)
            plt.scatter(my_data[x_label], predictions)
            #plt.grid()
            plt.show()

            predict_vec = np.vectorize(self.predict, signature='(n),(m)->()')
            predictions = predict_vec(my_data, x_opt)
            x_label = my_data.columns[0]
            plt.scatter(my_data[x_label], result_data)
            plt.scatter(my_data[x_label], predictions)
            #plt.grid()
            plt.show()


        return x_opt

    def eval(self, data):
        # for use after fitting has been done
        assert isinstance(data, (dict, pandas.DataFrame))
        assert self.x_opt is not None
        #input_data = [data[input] for input in self.input_signals]
        input_data = data[self.input_signals]
        result = self.predict_many(input_data, self.x_opt)
        return result

    @abstractmethod
    def verilog(self, lhs, coef_names):
        # return a list of strings
        # each string is a line of verilog
        # together, they should implement:
        # lhs = predict(self.input_signals, coef_names)
        pass

    @staticmethod
    def friendly_name(s):
        if isinstance(s, str):
            return s
        elif isinstance(s, (SignalIn, CenteredSignalIn, SignalArray)):
            return s.friendly_name()
        else:
            raise ValueError(f'Cannot find friendly name for {s}')

    def __str__(self):
        return self.name


class SympyExpression(Expression):
    def __init__(self, ast, io_symbols, coefs, name, bounds_dict={}):
        '''
        ast: sympy ast
        io_symbols: dict mapping from signals to symbols
        coefs: list of symbols
        name: name
        '''
        assert isinstance(io_symbols, dict)
        assert isinstance(coefs, list)
        self.ast = ast
        self.io_symbols = io_symbols
        self.input_signals = list(io_symbols.keys())
        self.coefs = coefs
        self.NUM_COEFFICIENTS = len(coefs)
        self.name = name
        self.bounds_dict = bounds_dict

        self.offset = 'should_be_set_in_recompile'
        self.nominal = 'should_be_set_in_recompile'

        self.recompile()

    def copy(self):
        # the reason for this is so that self.x_init is not shared
        # it's okay if they point to the same list, as long as replacing
        # the whole list isn't shared
        e = SympyExpression(self.ast, self.io_symbols, self.coefs, self.name,
                               bounds_dict=self.bounds_dict)
        e.x_init = self.x_init
        e.x_opt = self.x_opt
        return e

    def recompile(self):
        # recompile fun_compiled, recalculate offset and centered params

        # first, check that params make sense
        sympy_inputs = set(self.io_symbols.values()) | set(self.coefs)
        assert len(sympy_inputs) == len(self.io_symbols) + len(self.coefs)
        assert self.ast.free_symbols == sympy_inputs


        input_syms = [self.io_symbols[s] for s in self.input_signals]
        # I want to use cse=True, but there seems to be a bug in lambdify
        # so it doesn't always clean the special characters (angle brackets)
        # correctly when that option is turned on
        # Then I had a different bug that made me think sympy was at fault, so
        # I wrote this code to rename everything simply. That turned out to be
        # unnecessary, but now that I have that code I can use it and turn cse
        # back on again.
        #self.ast = self.ast.simplify()
        iss_renamed = []
        cs_renamed = []
        ast_renamed = self.ast
        count = 0
        for is_ in input_syms:
            is_renamed = Symbol(f'c{count}')
            count += 1
            iss_renamed.append(is_renamed)
            ast_renamed = ast_renamed.subs(is_, is_renamed)
        for c in self.coefs:
            c_renamed = Symbol(f'c{count}')
            count += 1
            cs_renamed.append(c_renamed)
            ast_renamed = ast_renamed.subs(c, c_renamed)
        self.fun_compiled = lambdify([iss_renamed, cs_renamed], ast_renamed, cse=True)

        # check for offset
        def is_offset(sym):
            # check if it's an offset:
            # One way is by taking a symbolic derivative, but
            # I'm worried about wasting effort here, but I actually don't have
            # a good way to do it otherwise ...
            # inspecting the ast is harder than
            # it seems because I had a case like (A*x + B*y) - A, so A looked
            # like an offset until I actually crawled the whole thing.
            # A cheap alternative might be a "numerical derivative" by checking
            # whether increasing the sym by d increases the result by d
            # Actually this only catches things with derivative exactly one,
            # while I should probably be catching anythign with constant
            # derivative. But then I'd have to store what the derivative is if
            # I ever want to do anything with the constant later, etc...
            #ot = Symbol('OFFSET_TEST')
            #test_a = self.ast.subs(sym, sym + ot)
            #test_b = self.ast + ot
            #return simplify(test_a - test_b) == 0
            # okaaay, that was a disaster because simplify is slow
            #d = diff(self.ast, sym)
            #return d == 1
            # and diff is slow. great.
            if not self.ast.is_Add:
                return False
            appears_as_const = False
            appears_elsewhere = False
            for arg in self.ast.args:
                if arg == sym:
                    appears_as_const = True
                elif sym in arg.free_symbols:
                    appears_elsewhere = True
            return appears_as_const and not appears_elsewhere

        self.offset = None
        for sym in self.ast.free_symbols:
            if is_offset(sym):
                #if self.offset is not None:
                #    print(f'Warning: multiple offset coefficients for "{self.name}" expr "{self.ast}", "{self.offset}" and "{sym}"')
                self.offset = sym

        # check for centering
        self.nominal = self.ast
        for input_signal, input_symbol in self.io_symbols.items():
            if input_signal.nominal is None:
                # this is not a function of optional inputs
                self.nominal = None
                break
            self.nominal = self.nominal.subs(input_symbol, input_signal.nominal)


    def predict(self, opt_values, coefs):
        # lambdify should be smart enough to accept the input in list form
        # or not
        ans = self.fun_compiled(opt_values, coefs)
        return ans

    def predict_many(self, opt_value_data, coefs):
        ans = self.fun_compiled(opt_value_data.values.T, coefs)
        if opt_value_data.shape[1] == 0:
            # fun_compiled will not respect shape[0] in this case
            ans = np.zeros(opt_value_data.shape[0]) + ans
        return ans


    def verilog(self, lhs, coef_names):
        assert isinstance(lhs, str)
        coef_subs = zip(self.coefs, [Symbol(c) for c in coef_names])
        ast_coef_names = self.ast.subs(coef_subs)
        return [lhs + '=' + str(ast_coef_names)]

    def vector_cleanup(self, signal, new_signals, new_signals_sym, old_coefs, new_coefs):
        assert isinstance(signal, Signal)

        del self.io_symbols[signal]
        for new_signal, new_signal_sym in zip(new_signals, new_signals_sym):
            self.io_symbols[new_signal] = new_signal_sym
        self.input_signals.remove(signal)
        self.input_signals += new_signals

        for old_coef in old_coefs:
            self.coefs.remove(old_coef)
        self.coefs += new_coefs
        self.coefs.sort(key=lambda sym: sym.name)
        self.NUM_COEFFICIENTS = len(self.coefs)

        self.recompile()
        return


    def vector_fallback(self, signal, new_signals, signal_sym, new_signals_sym):
        '''
        Vector the signal by replacing it with a dot product of its
        components and new weight coefficients.
        self.vector() will do something better and try to absorb an existing
        scaling coefficient into the weights, but if that fails it will call
        this function as a fallback.
        '''
        # TODO we want to constrain weights so that magnitude is 1
        # I think I will just do all but one and calculate the last one
        # I wish it were easier in my system to assign bounds and/or constraints
        new_coefs_sym = [Symbol(f'{s.friendly_name()}_weight_{self.name}')
                       for s in new_signals]

        sym_vectored = Add(Mul(c, s)
                           for c, s in zip(new_coefs_sym, new_signals_sym))
        self.ast = self.ast.subs((signal_sym, sym_vectored))
        self.vector_cleanup(signal, new_signals, new_signals_sym,
                            [], new_coefs_sym)

    def vector(self, signal, new_signals):
        '''
        Replace signal with new_signals, editing self.ast accordingly
        Edits: NUM_COEFFICIENTS, ast, coefs, input_signals, io_symbols,
            compiled_fun
        Our goal is to replace signal with (A*compA + B*compB), where
        A^2+B^2=1. But ideally we parameterize that with only one coef.
        For 2 dimensions the coef can be tanB/A), but for more dimensions
        I dn't know a good parameterization.
        ALSO: in cases where every instance of signal has a coefficien, I would
        like to absordb that into the A and B.
        So gain*in -> gainA*inA + gainB*inB.
        TODO what if the thing has signal^2
        '''

        assert signal in self.input_signals
        signal_sym = self.io_symbols[signal]
        new_signals_sym = [Symbol(s.friendly_name()) for s in new_signals]

        all_new_coefs = []

        # this should be an integer, but I need a pointer to an integer to do
        # some globalish variable hack
        fallback_name_count = [0]
        def fallback_expr():
            num = len(new_signals) - 1
            syms = [Symbol(f'{signal.friendly_name()}_vec{i}')
                    for i in range(fallback_name_count[0], fallback_name_count[0]+num)]
            fallback_name_count[0] += num
            for sym in syms:
                all_new_coefs.append(sym)
                self.bounds_dict[sym] = (-1, 1)

            size = '+'.join(f'{sym.name}**2' for sym in syms)
            final_str = f'sqrt(1-({size}))'
            final = sympy.parse_expr(final_str)
            ast = Add(*[Mul(c, s) for c, s in zip(syms+[final], new_signals_sym)])
            return ast

        def replace_one_signal(ast, sub):
            # walk the ast and subs the first occurrence of signal_sym you find
            if ast == signal_sym:
                return sub
            else:
                done = False
                new_args = []
                for arg in ast.args:
                    if done:
                        new_args.append(arg)
                    elif signal_sym in arg.free_symbols:
                        new_arg = replace_one_signal(arg, sub)
                        new_args.append(new_arg)
                        done = True
                    else:
                        new_args.append(arg)
                assert done, f'Failed to replace_one_signal, could not find {signal_sym} in {ast}'
                ans = ast.func(*new_args)
                return ans




        def vector_coef(coef_to_vector):
            new_coefs = [Symbol(f'{coef_to_vector.name}_{s.friendly_name()}')
                         for s in new_signals]
            for c in new_coefs:
                all_new_coefs.append(c)

            new_expr = Add(
                *[Mul(c, s) for c, s in zip(new_coefs, new_signals_sym)])
            return new_expr

        def combine_coefs(distributed_coef, original_coef):
            n = f'{distributed_coef.name}_times_{original_coef.name}'
            c = Symbol(n)
            all_new_coefs.append(c)
            return c


        def distribute(coef, tree):
            # distribute coef into tree[0] according to path in tree
            ast, children = tree
            if len(children) > 0:
                processed_children = [distribute(coef, c) for c in tree[1]]
                return Add(*processed_children)
            else:
                if ast == signal_sym:
                    # this is where we vector the input with a coef
                    new_ast = vector_coef(coef)
                    return new_ast
                elif ast in self.coefs:
                    # combine two coefs into one
                    return combine_coefs(coef, ast)
                else:
                    # if it doesn't fit one of the above cases I don't think
                    # it should have been in the tree
                    assert False, f'Internal error in equation vectoring for {self.ast}'

        def vec(ast):
            '''
            If the input is not here at all, or you can't distribute into this
            ast and combine with the input, return None.
            If it would be okay to distribute a coefficient into this ast,
            return a list of child asts to distribute into

            Different states we might want to return:
            0: This ast would like a coefficient
            1: This ast is a coefficient
            2: This ast could take a coefficient into it
            3: This ast could not take a coefficient

            In all cases, return a tuple with the case and then additional stuff
            Case 0: return a list of tree nodes to multiply the coef into
                The tree nodes are tuples of (ast, [child nodes])

            NOTE: I'm being a bit lazy about cases where there are multiple
            things to be distributed. I may fall back when I don't need to.
            For example, a*b*(in-c)*(in-d), I'll only end up distributing the
            a into the first parenthetical, and won't distribute the b into the
            second even though I should.
            Additionally, in a*b*(c + (in-d)*(in-e)) I could distribute both
            of them in but won't. This example is much harder to fix than the
            last because I'd have to keep track of how many coefficients each
            subtree could take. I'm not gonna bother.
            '''

            if ast in self.coefs:
                return 1, (ast, [])

            elif ast == signal_sym:
                return 0, (ast, [])

            elif ast.is_Add:
                children = [vec(a) for a in ast.args]
                cases = [c[0] for c in children]
                if 0 in cases:
                    if 3 in cases:
                        # we need to fall back, and this is case 3
                        assert False
                    else:
                        # will be case 0
                        new_ast = Add(*list(c[1][0] for c in children))
                        return 0, (new_ast, [c[1]
                                         for a, c in zip(ast.args, children)])
                else:
                    new_ast = Add(*list(c[1][0] for c in children))
                    if 3 in cases:
                        # will be case 3
                        return 3, (new_ast, )
                    else:
                        # will be case 2
                        return 2, (new_ast, [c[1]
                                         for a, c in zip(ast.args, children)])

            elif ast.is_Mul:
                children = [vec(a) for a in ast.args]
                cases = [c[0] for c in children]
                if 0 in cases:
                    if 1 in cases:
                        # Nice! we can put a coefficient in
                        i0 = cases.index(0)
                        i1 = cases.index(1)
                        result = distribute(children[i1][1][0], children[i0][1])

                        # I'm theoretically missing opportunities to do further
                        # vectoring with other instances of the input, but I
                        # am not going to worry about it

                        remaining = [c[1][0] for i, c in enumerate(children)
                                     if i not in [i0, i1]]
                        result = Mul(*remaining, result)
                        return 3, (result, )
                    else:
                        # this will be a case 0
                        assert False
                else:
                    new_ast = Mul(*list(c[1][0] for c in children))
                    if 1 in cases:
                        # cool, we can distribute into the coef
                        i = cases.index(1)
                        return 2, (new_ast, [children[i][1]])
                    elif 2 in cases:
                        # still okay, we can distribute into something else
                        i = cases.index(2)
                        return 2, (new_ast, [children[i][1]])
                    else:
                        # no way to take a coef, this is case 3
                        return 3, (new_ast, )

            else:
                # we don't know how to deal with this; it's case 3
                if len(ast.args) > 0:
                    children = [vec(a) for a in ast.args]
                    new_args = [c[1][0] for c in children]
                    new_ast = ast.func(*new_args)
                else:
                    new_ast = ast
                return 3, (new_ast, )


        case, tree = vec(self.ast)
        self.ast = tree[0]


        # Fallback replacement

        while signal_sym in self.ast.free_symbols:
            # still another occurrence that was not vectored
            fallback = fallback_expr()
            self.ast = replace_one_signal(self.ast, fallback)


        old_coefs = [c for c in self.coefs if c not in self.ast.free_symbols]
        self.vector_cleanup(signal, new_signals, new_signals_sym, old_coefs, all_new_coefs)
        return







def get_ast_from_signal(signal, qualifier=None):
    if isinstance(signal, SignalIn):
        # centered version of signal
        assert isinstance(signal, SignalIn)
        sym = Symbol(signal.friendly_name())
        ast = sym - signal.nominal
        return ast, {signal: sym}, []
    elif isinstance(signal, SignalArray):
        # weighted sum of bits
        assert isinstance(signal, SignalArray)
        assert len(signal.shape) == 1, 'TODO multidimensional optional input'
        bits = list(signal)
        if qualifier is None:
            weights = [Symbol(f'{b.friendly_name()}{qualifier}_w{i}') for i, b in
                       enumerate(bits)]
        else:
            weights = [Symbol(f'{qualifier}_w{i}') for i, b in enumerate(bits)]
        bits_sym = [Symbol(b.friendly_name()) for b in bits]
        ast = sum(b * w for b, w in zip(bits_sym, weights))
        bits_dict = {b: s for b, s in zip(bits, bits_sym)}
        return ast, bits_dict, weights
    else:
        assert False


def get_default_expr_from_signal(signal, name):
    # Does not have an offset coefficient
    # Already centered, so expr(nominal) = 0
    if isinstance(signal, SignalArray):
        ast, s_syms, c_syms = get_ast_from_signal(signal, name)
        nominal = signal.get_binary_value(signal.nominal)
        # NOTE next line depends on implementation of get_ast_from_signalarray
        nominal_ast = sum(c*n for c, n in zip(c_syms, nominal))
        ast_centered = ast - nominal_ast
        expr = SympyExpression(ast_centered, s_syms, c_syms, name)
        return expr
    elif isinstance(signal, SignalIn):
        assert isinstance(signal.value, (list, tuple)), f'Trying to model as a function of {signal}, but it has pinned value {signal.value} instead of a range'
        assert len(signal.value) == 2, f'Trying to model as a function of {signal}, but it has value {signal.value} instead of a range'
        ast, s_syms, c_syms = get_ast_from_signal(signal)
        w_sym = Symbol(f'{name}_w')
        ast_weighted = w_sym * ast
        expr = SympyExpression(ast_weighted, s_syms, c_syms + [w_sym], name)
        return expr
    else:
        assert False, f'Cannot create ast for signal {signal} of type {type(signal)}'


class ConstExpression(SympyExpression):
    def __init__(self, name):
        sym = Symbol(name)
        super().__init__(sym, {}, [sym], name)

    def copy(self):
        e = ConstExpression(self.name)
        e.x_init = self.x_init
        e.x_opt = self.x_opt
        return e


# TODO fix the spelling lol
class HierarchicalExpression(SympyExpression):
    '''
    The most confusing thing about the HierarchicalExpression is the way it
    manages the offsets of child expressions.  I'm changing it so that the
    child expressions can't have offsets. Every optional influence must have
    no offset, and it must evaluate to zero when the optional inputs are
    nominal. I think this is doable by changing the expressions themselves,
    and I think it's easier to explain to the end user this way.
    '''
    def __init__(self, parent_expression, child_expressions, name,
                 bounds_dict={}):
        # TODO this never calls super().__init__() and it probably should
        # each coefficient in the parent expression is the result of evaluating
        # a child expression
        # child_expression_names should match the parent inputs
        self.name = name
        self.manage_offsets = False #isinstance(parent_expression, SumExpression)
        assert isinstance(child_expressions, dict)
        assert all(isinstance(k, Symbol) for k in child_expressions)
        assert all(isinstance(v, Expression) for v in child_expressions.values())

        # TODO I think there could be a bug if we are managing coefficients
        #  but one of the child expressions is a ConstExpression
        assert parent_expression.NUM_COEFFICIENTS == (len(child_expressions) + (1 if self.manage_offsets else 0))
        self.parent_expression = parent_expression
        self.child_expressions = child_expressions

        child_inputs = [s for ce in child_expressions.values() for s in ce.input_signals]
        self.input_signals = list(parent_expression.input_signals) + child_inputs

        # compile our own ast

        io_symbols = self.parent_expression.io_symbols.copy()
        coefs = [c for c in self.parent_expression.coefs
                 if c not in child_expressions]
        assert isinstance(parent_expression, SympyExpression)
        if self.manage_offsets:
            assert False
            child_asts = []
            for child in self.child_expressions:
                if isinstance(child, ConstExpression):
                    child_asts.append(0)
                    # no coef
                    continue
                if not child.last_coef_is_offset:
                    child_asts.append(child.ast)
                    io_symbols.update(child.io_symbols)
                    coefs += child.coefs
                    continue

                child_offset_sym = child.coefs[-1]
                child_ast_removed_offset = child.ast.subs(child_offset_sym, 0)
                child_asts.append(child_ast_removed_offset)
                io_symbols.update(child.io_symbols)
                coefs += child.coefs[:-1]

        else:
            for sym, child in self.child_expressions.items():
                io_symbols.update(child.io_symbols)
                coefs += child.coefs

        final_ast = self.parent_expression.ast
        for sym, child_expr in self.child_expressions.items():
            final_ast = final_ast.subs(sym, child_expr.ast)

        self.ast = final_ast
        self.io_symbols = io_symbols
        self.coefs = sorted(set(coefs), key=lambda c: c.name)
        #assert len(self.coefs) == self.NUM_COEFFICIENTS
        self.NUM_COEFFICIENTS = len(self.coefs)

        # self.inputs probably has repeats,
        # but self.io_symbols.keys() should be the same set without repeats
        assert set(self.input_signals) == set(self.io_symbols)
        self.input_signals = sorted(self.io_symbols.keys(),
                                    key=lambda s: s.friendly_name())
        input_syms = [self.io_symbols[s] for s in self.input_signals]

        for sym in self.ast.free_symbols:
            assert sym in input_syms or sym in self.coefs

        self.bounds_dict = bounds_dict

        self.recompile()

    def copy(self):
        e = HierarchicalExpression(
            self.parent_expression.copy(),
            {lhs: e.copy() for lhs, e in self.child_expressions.items()},
            self.name,
            self.bounds_dict
        )
        e.x_init = self.x_init
        e.x_opt = self.x_opt
        return e







    def fit_by_group(self, optional_data, result_data, mode_prefix):
        # relies on a specific structure of heirarchy and specific sweep groups
        # in optional_data in order to fit children and grandchildren one
        # piece at a time

        # TODO we should update PlotHelper to make this case, where we only
        #  use if for mode_prefix, more intentional
        ph = PlotHelper(None, None, mode_prefix, None, None)

        self.ces_ordered = list(self.child_expressions.values())
        for child in self.ces_ordered:
            assert child.parent_expression.ast.is_Add or child.parent_expression.ast.is_Symbol
            ces = list(child.child_expressions.values())
            const_loc = None
            for i, ce in enumerate(ces):
                if isinstance(ce, ConstExpression):
                    assert const_loc is None
                    const_loc = i
            assert const_loc is not None
            ces[0], ces[const_loc] = ces[const_loc], ces[0]
            child.ces_ordered = ces

            assert isinstance(child.ces_ordered[0], ConstExpression)
            for grandchild in child.ces_ordered[1:]:
                assert grandchild.offset is None
                assert grandchild.nominal == 0


        # TODO I'm still getting nan instead of None for sg sometimes
        groups = {sg for sg in optional_data[SampleManager.GROUP_ID_TAG]
                  if (sg is not None and not isinstance(sg, float))}
        groups = sorted(groups, key=lambda sg: sg.name)

        # first thing is to figure out which sweep groups to use for which
        # fit expressions
        fits_for_sweeps = defaultdict(list)
        #for i in range(len(self.ces_ordered)):
        #    child = self.ces_ordered[i]

        # kind of probelmatic

        for child in self.ces_ordered:
            for grandchild in child.ces_ordered:
                if isinstance(grandchild, ConstExpression):
                    continue
                # which sweep is best for grandchild.input_signals
                # first, search for an exact match
                def break_buses(ss):
                    # break sample groups that are buses into their bits
                    sg_signals = []
                    for s in ss:
                        if isinstance(s, SignalArray):
                            assert len(s.array.shape) == 1, 'TODO nested buses'
                            sg_signals += list(s)
                        else:
                            sg_signals.append(s)
                    return sg_signals
                perfect_match_found = False
                for sg in groups:
                    if (set(break_buses(grandchild.input_signals))
                            == set(break_buses(sg.signals))):
                        # perfect match, use this sweep group
                        fits_for_sweeps[sg].append(grandchild)
                        perfect_match_found = True

                # TODO this will include all partial matches, but in reality
                #  I think we should rank them and include only the best
                if not perfect_match_found:
                    # no perfect match found
                    # partial match is when the sweep group has extra things
                    for sg in groups:
                        if all(s in break_buses(sg.signals)
                               for s in break_buses(grandchild.input_signals)):
                            # all the things in the fit expr were swept
                            fits_for_sweeps[sg].append(grandchild)

        # now we loop through each SampleGroup
        result_fits = defaultdict(list)
        for sg in groups:
            # group_data is everything during the sweep of sg
            group_data_indices = optional_data[SampleManager.GROUP_ID_TAG] == sg
            group_data = optional_data[group_data_indices]
            group_data_res = result_data[group_data_indices]

            def get_data_for_id(si):
                point_indices = group_data[SampleManager.SWEEP_ID_TAG] == si
                point_data = group_data[point_indices]
                example_row_id = point_indices.idxmax()
                point_data_res = group_data_res[point_indices]

                return point_data, point_data_res, example_row_id

                #self.parent_expression.fit(point_data, point_data_res,
                #                           parent_by_group_x_init)
                #result = self.parent_expression.x_opt
                #error = self.parent_expression.error(
                #    point_data[self.parent_expression.input_signals],
                #    self.parent_expression.x_opt,
                #    point_data_res)
                #return result, error


            # First, we find values for each param, with optional inputs fixed
            sweep_ids = {si for si in group_data[SampleManager.SWEEP_ID_TAG]}
            group_results = []
            # example rows are one row from each sweep point
            example_row_ids = []
            plot_xss = []
            plot_ys = []
            plot_predictions = []
            worst_example = (-1, 0)
            parent_by_group_x_init = None
            # do a first pass for each one to hopefully settle on a good guess
            for si in sorted(sweep_ids):
                point_data, point_data_res, example_row_id = get_data_for_id(si)
                self.parent_expression.fit(point_data, point_data_res,
                                           parent_by_group_x_init)
                parent_by_group_x_init = self.parent_expression.x_opt
            # now fit them all a second time since we should have a good
            # initial guess from the previous time around
            for si in sorted(sweep_ids):
                # point_data is the data from one point (si) on the sg sweep
                #point_indices = group_data[SampleManager.SWEEP_ID_TAG] == si
                #point_data = group_data[point_indices]
                #example_row_ids.append(point_indices.idxmax())
                #point_data_res = group_data_res[point_indices]

                #self.parent_expression.fit(point_data, point_data_res,
                #                           parent_by_group_x_init)

                point_data, point_data_res, example_row_id = get_data_for_id(si)
                self.parent_expression.fit(point_data, point_data_res,
                                           parent_by_group_x_init)

                #print(f'Result for group={sg}, sweep_id={si}:')
                #print(self.parent_expression.coefs)
                #print(self.parent_expression.x_opt)
                #print()

                parent_by_group_x_init = self.parent_expression.x_opt
                group_results.append(self.parent_expression.x_opt)
                example_row_ids.append(example_row_id)

                error = self.parent_expression.error(
                     point_data[self.parent_expression.input_signals],
                     self.parent_expression.x_opt,
                     point_data_res)
                if error > worst_example[1]:
                    worst_example = (si, error)

                plot_xss.append(point_data)
                plot_ys.append(point_data_res)
                #predict_data = [point_data[input] for input in self.parent_expression.input_signals]
                predict_data = point_data[self.parent_expression.input_signals]
                predictions = self.parent_expression.predict_many(predict_data, self.parent_expression.x_opt)
                plot_predictions.append(predictions)
            group_results = np.array(group_results)

            if PLOT:
                # plotting initial small plots
                for xaxis in self.parent_expression.input_signals:
                    if xaxis == fixture.regression.Regression.one_literal:
                        # TODO if this is the only xaxis, should we not skip?
                        continue

                    #print('New figure for ', self.name, sg)
                    plt.figure()
                    colors = []
                    orders = []
                    for xss, ys, preds in zip(plot_xss, plot_ys, plot_predictions):
                        xs = xss[xaxis]
                        order = np.argsort(xs)
                        orders.append(order)
                        temp = plt.plot(np.array(xs)[order], np.array(ys)[order], '*')[0]
                        colors.append(temp.get_color())

                    for xss, preds, c, order in zip(plot_xss,
                                             plot_predictions, colors, orders):
                        xs = xss[xaxis]
                        plt.plot(np.array(xs)[order], np.array(preds)[order], '--', color=c)

                    legend = []
                    MAX_LEGEND_LEN = 16
                    def sg_vals_str(xss):
                        vals = []
                        for s in sg.signals:
                            v = list(xss[s])[0]
                            v_str = str(v) if isinstance(v, int) else f'{v:.4g}'
                            vals.append(f'{s.friendly_name()}={v_str}')
                        return ', '.join(vals)

                    for xss in plot_xss:
                        if len(legend) < MAX_LEGEND_LEN:
                            legend.append(sg_vals_str(xss))

                    plt.legend(legend)
                    plt.title(f'Fitting {self.name} for various {sg}')
                    plt.xlabel(f'{xaxis.friendly_name()}')
                    plt.ylabel(f'{self.parent_expression.name}')
                    ph._save_current_plot(f'individual_fits/{self.name}/{sg.name}/Fits for {self.name} vs {xaxis.friendly_name()} from Sweeping {sg.name}')

                    # now plot the worst one by itself
                    plt.figure()
                    worst_i = sorted(sweep_ids).index(worst_example[0])
                    xss = plot_xss[worst_i]
                    xs = xss[xaxis]
                    ys = plot_ys[worst_i]
                    preds = plot_predictions[worst_i]
                    sg_val = sg_vals_str(get_data_for_id(worst_example[0])[0])

                    order = np.argsort(xs)
                    plt.plot(np.array(xs)[order], np.array(ys)[order], '*')
                    plt.plot(np.array(xs)[order], np.array(preds)[order], '--')
                    plt.xlabel(f'{xaxis.friendly_name()}')
                    plt.ylabel(f'{self.parent_expression.name}')
                    plt.legend(['Measured', 'Predicted'])
                    plt.title(f'Fitting {self.name}, worst fit at {sg_val}')
                    ph._save_current_plot(f'individual_fits/{self.name}/{sg.name}/Worst fit for {self.name} vs {xaxis.friendly_name()} from Sweeping {sg.name}')


            # Next we find expressions for each param, using values from earlier
            # each row in example_data corresponds to one point in the sweep
            # TODO I'd like to delete the columns corresponding to test inputs
            #  in example_data because they are not relevant. But I don't have
            #  a good way to find the names for those. It's just confusing for
            #  somebody maintaining the code
            example_data = group_data.loc[example_row_ids]
            for i in range(len(self.ces_ordered)):
                child = self.ces_ordered[i]
                child_results = group_results[:, i]

                # When we fit by group we don't exactly follow the normal
                # hierarchy for the child, so it has to be in a specific form
                # TODO move this assertion to the top?
                assert isinstance(child, HierarchicalExpression)
                assert isinstance(child.parent_expression, SumExpression)

                ## now we look through child's children and determine which
                # one(s) are actually affected by changing sg
                # TODO sg.signal doesn't always exist?
                # TODO what about when sg has multiple signals?
                # TODO what about when child.input_signals has multiple signals?
                #relevant_grandchildren = []
                for grandchild in child.ces_ordered:
                    if grandchild not in fits_for_sweeps[sg]:
                        continue
                    # we are here because this grandchild actually depends on
                    # the sg we are currently working with
                    offset_sym = Symbol('grandchild_offset')
                    grandchild_with_offset = SympyExpression(
                        grandchild.ast + offset_sym,
                        grandchild.io_symbols,
                        grandchild.coefs + [offset_sym],
                        f'{grandchild.name}_individualfit'
                    )
                    grandchild_with_offset.fit(example_data, child_results)
                    print(f'Result for sg={sg} i={i}:')
                    print(grandchild_with_offset.coefs)
                    print(grandchild_with_offset.x_opt)
                    print()

                    if PLOT:
                        # plotting secondary fits, from using fit params as goals
                        plt.figure()
                        # TODO this may not work with future sg types
                        xaxis = sg.signals[-1]
                        xs = example_data[xaxis]
                        #xs_sampler = fixture.sampler.get_sampler_for_signal(xaxis)
                        x_targets = np.linspace(0, 1, 100).reshape((100,1))
                        targets = np.concatenate((x_targets, 0.5*np.ones((100, sg.NUM_DIMS-1))), 1)
                        data_smooth = pandas.DataFrame(sg.get_many(targets))
                        data_smooth = data_smooth.sort_values(by=[xaxis])
                        predictions = grandchild_with_offset.predict_from_dict(data_smooth, grandchild_with_offset.x_opt)
                        xs_smooth = data_smooth[xaxis]

                        xs_order = np.argsort(xs)
                        plt.plot(np.array(xs)[xs_order], child_results[xs_order], '*')
                        plt.plot(xs_smooth, predictions, 'x--')
                        plt.legend(['Measured', 'Predicted'])
                        plt.xlabel(f'{xaxis.friendly_name()}')
                        plt.ylabel(f'{child.name}')
                        #plt.ylim((0, 3000))
                        plt.title(f'{grandchild_with_offset.name} vs. {sg}')
                        ph._save_current_plot(f'individual_fits/{self.name}/{sg.name}/Individual fit for {grandchild_with_offset.name} vs {sg}')

                        # TEMP for checking x_init
                        if grandchild_with_offset.x_init is not None:
                            plt.figure()
                            predictions = grandchild_with_offset.predict_from_dict(data_smooth, grandchild_with_offset.x_init)
                            #plt.plot(xs, child_results, '*')
                            plt.plot(np.array(xs)[xs_order],
                                     child_results[xs_order], '*')
                            plt.plot(xs_smooth, predictions, 'x--')
                            plt.legend(['Measured', 'Predicted'])
                            plt.xlabel(f'{xaxis.friendly_name()}')
                            plt.ylabel(f'{child.name}')
                            plt.title(f'Initial Minimizer Guess for\n{grandchild_with_offset.name} vs. {sg}')
                            ph._save_current_plot(f'individual_fits/{self.name}/{sg.name}/debug/Initial point for minimizer for {grandchild_with_offset.name} vs {sg}')
                        print()

                    result_fits[child.ces_ordered[0]].append(grandchild_with_offset.x_opt[-1:])
                    result_fits[grandchild].append(grandchild_with_offset.x_opt[:-1])

        # I think that result_fits should have one entry per optional input
        # expression, plus one entry per optional input for the constants
        # With proper centering, the const fits should all be equal, so we
        # just use their average
        # TODO I'm not sure it's wise to directly edit grandchild.x_init,
        #  but I think since the objects won't be used to fit again it's
        #  probably okay
        # TODO redundant offsets?
        for grandchild, fits in result_fits.items():
            x_init = sum(fits) / len(fits)
            grandchild.x_init = x_init


        def scatter(coefs, title):
            predict_data = optional_data[self.input_signals]
            predictions = self.predict_many(predict_data, coefs)
            plt.scatter(result_data, predictions)
            plt.title(title)
            plt.xlabel('Measured')
            plt.ylabel('Predicted')

            ph._save_current_plot(
                f'individual_fits/{self.name}/debug/{title}')

        if self.x_init is not None:
            scatter(self.x_init, f'Initial Fit for {self.name}')

        # now we are ready
        self.fit(optional_data, result_data)

        scatter(self.x_opt, f'Result for {self.name}')

        if False and PLOT:

            #predict_data = [optional_data[col] for col in self.input_signals]
            predict_data = optional_data[self.input_signals]
            predictions = self.predict_many(predict_data, self.x_opt)
            if len(self.parent_expression.input_signals) == 0:
                xaxis = list(range(len(predictions)))
            elif len(self.parent_expression.input_signals) == 1:
                xaxis = self.parent_expression.input_signals[0]
            else:
                # TODO fix this xaxis
                xaxis = self.parent_expression.input_signals[0]
            xs = optional_data[xaxis]
            plt.figure()
            plt.plot(xs, result_data, '*')
            plt.plot(xs, predictions, '--')
            #plt.grid()
            plt.xlabel(f'{xaxis.friendly_name()}')
            plt.ylabel(f'{self.name}')
            plt.show()
        print()

    def predict(self, opt_values, coefs):
        # first, use the child expressions to find each param
        assert len(coefs) == self.NUM_COEFFICIENTS
        assert len(opt_values) == len(self.input_signals)
        coef_count = 0
        parent_coefs = []
        for ce in self.child_expressions:
            input_indices_ce = [self.input_signals.index(s) for s in ce.input_signals]
            input_values_ce = [opt_values[i] for i in input_indices_ce]

            num_child_coefficients = ce.NUM_COEFFICIENTS
            coefs_ce = coefs[coef_count : coef_count + num_child_coefficients]
            coef_count += num_child_coefficients

            parent_coef = ce.predict(input_values_ce, coefs_ce)
            parent_coefs.append(parent_coef)

        assert coef_count == len(coefs)

        parent_input_indices = [self.input_signals.index(s) for s in
                            self.parent_expression.input_signals]
        parent_input_values = [opt_values[i] for i in parent_input_indices]
        ans = self.parent_expression.predict(parent_input_values, parent_coefs)
        return ans


    @property
    def x_opt(self):
        return self._x_opt
    @x_opt.setter
    def x_opt(self, x_opt):
        assert len(x_opt) == len(self.coefs)
        results = {c: v for c, v in zip(self.coefs, x_opt)}
        for child in self.child_expressions.values():
            child_x_opt = [results[c] for c in child.coefs]
            child.x_opt = child_x_opt
        self._x_opt = x_opt
        return

        # set self.opt for self and all the children
        # the incoming x_opt is short because it has only the aggregate offset
        # the final x_opt_new is long because is has the offset for each child
        # and also the aggregate offset (shifted to account for the child
        # offsets)
        x_opt_count = 0
        x_opt_new = x_opt.copy()
        x_opt_new_count = 0
        for ce in self.child_expressions:
            slice_length = ce.NUM_COEFFICIENTS
            if self.manage_offsets and ce.last_coef_is_offset:
                slice_length -= 1
            x_opt_ce = x_opt[x_opt_count : x_opt_count + slice_length]
            x_opt_count += slice_length
            x_opt_new_count += slice_length
            if self.manage_offsets and ce.last_coef_is_offset:
                ce_inputs_nom = [s.nominal for s in ce.input_signals]
                x_opt_ce_zero = np.concatenate((x_opt_ce, [0]))
                offset_nom = -1 * ce.predict(ce_inputs_nom, x_opt_ce_zero)
                x_opt_ce = np.concatenate((x_opt_ce, [offset_nom]))
                x_opt_new = np.insert(x_opt_new, x_opt_new_count, offset_nom)
                x_opt_new_count += 1
            ce.x_opt = x_opt_ce

        assert x_opt_count == len(x_opt)
        assert x_opt_new_count == len(x_opt_new)

        # we do this at the end because it can get edited with managed offsets
        self._x_opt = x_opt_new

    @property
    def x_init(self):
        # TODO error handling
        result_dict = {}
        for child in self.child_expressions.values():
            if child.x_init is None:
                return None
            assert len(child.coefs) == len(child.x_init)
            for c, v in zip(child.coefs, child.x_init):
                result_dict[c] = v
        x_init = [result_dict[c] for c in self.coefs]
        return x_init

    @x_init.setter
    def x_init(self, x_init):
        if x_init is None:
            self.parent_expression.x_init = None
            for c in self.child_expressions.values():
                c.x_init = None
            return
        if not all(isinstance(c, ConstExpression)
                   for c in self.child_expressions):
            assert len(x_init) == len(self.coefs)
            results = {c: v for c, v in zip(self.coefs, x_init)}
            nominal_parent_values = {}
            for child in self.child_expressions.values():
                child_x_init = [results[c] for c in child.coefs]
                child.x_init = child_x_init

                if (isinstance(child, HierarchicalExpression)
                        and isinstance(child.parent_expression, SumExpression)):
                    # if any of this child's children are const, then they
                    # probably represent a nominal value for one of the parents
                    for grandchild in child.child_expressions.values():
                        if isinstance(grandchild, ConstExpression):
                            nominal_parent_values[child] = grandchild.x_init[0]

            if len(nominal_parent_values) > 0:
                p = self.parent_expression
                # we should set the parent x_init as well
                p.x_init = np.zeros(len(p.coefs))
                for c, nominal in nominal_parent_values.items():
                    index = [c.name for c in p.coefs].index(c.name)
                    p.x_init[index] = nominal

            return
        else:
            assert False, 'TODO: unexpected case'


    def _search(self, target_name):
        # return a list of child expressions with the target name
        matches = []
        if self.name == target_name:
            matches.append(self)

        for child in self.child_expressions:
            if isinstance(child, HierarchicalExpression):
                matches += child._search(target_name)
            else:
                if child.name == target_name:
                    matches.append(child)
        return matches

    def search(self, target_name):
        # return a child expression with the target_name, or None
        matches = self._search(target_name)
        if len(matches) == 0:
            return None
        elif len(matches) > 1:
            raise KeyError(f'Found multiple Expressions matching "{target_name}" in "{self.name}"')
        else:
            return matches[0]


    def verilog(self, lhs, coef_names):
        assert len(coef_names) == self.NUM_COEFFICIENTS
        coef_name_map = {coef: name for coef, name in zip(self.coefs, coef_names)}
        def name(expr):
            # seems like expr.name is already qualified with self.name
            return f'{expr.name}'
        name_nominal = f'{self.name}_nominal'
        lines = []

        # parent expression first
        parent_coef_names = [name(ce) for ce in self.child_expressions.values()]
        if self.manage_offsets:
            parent_coef_names.append(name_nominal)
        #assert len(parent_coef_names) == self.parent_expression.NUM_COEFFICIENTS
        lhs_str = lhs if isinstance(lhs, str) else lhs.friendly_name()
        lines += self.parent_expression.verilog(lhs_str, parent_coef_names)

        # all the child expressions
        for ce in self.child_expressions.values():
            child_coefs = [coef_name_map[c] for c in ce.coefs]
            lines += ce.verilog(name(ce), child_coefs)
        return lines

    def recompile(self):
        # just check that things make sense and then do the normal recompile
        input_symbols = set(self.io_symbols.values())
        for parent_sym in self.parent_expression.ast.free_symbols:
            assert (parent_sym in input_symbols
                    or parent_sym in self.child_expressions
                    or parent_sym in self.coefs)

        super().recompile()


class SumExpression(SympyExpression):
    input_signals = []

    def __init__(self, n, name):
        io_symbols = {}
        coefs = [Symbol(f'{name}_c{i}') for i in range(n)]
        ast = sum(coefs)
        super().__init__(ast, io_symbols, coefs, name)

    def copy(self):
        e = SumExpression(len(self.coefs), self.name)
        e.x_init = self.x_init
        e.x_opt = self.x_opt
        return e

    def verilog(self, lhs, coef_names):
        opt_names = [self.friendly_name(s) for s in self.input_signals]
        assert len(opt_names) == 0
        assert len(coef_names) == self.NUM_COEFFICIENTS
        ans = ' + '.join(f'{cn}' for cn in coef_names)
        return [f'{lhs} = {ans};']



def get_ast_from_string(string):
    # some builtin functions (like abs) we want to keep as functions,
    # but others (like input) we want to not do that

    def should_tag(s):
        # things that are builtin/keyword names, but are more likely to be a
        # circuit pin than a sympy algebra keyword
        # Some of the boolean names like 'and' 'or' 'not' are kinda problematic,
        # although I think sympy typically has you use '&' '|' '!' instead
        # originally I was using iskeyword() in this function as well
        should = ['in', 'input', 'raise', 'from', 'assert', 'and', 'or', 'not']
        return s in should




    FIXTURE_TAG = 'FIXTURE_TAG_'
    def transform_name(tokens, local_dict, global_dict):
        result = []

        for type_, name in tokens:
            if type_ == 1 and should_tag(name):
                result.append((1, FIXTURE_TAG + name))
            else:
                result.append((type_, name))
        return result

    def untransform_name(tokens, local_dict, global_dict):
        result = []
        for type_, name in tokens:
            if type_ == 1 and FIXTURE_TAG in name:
                result.append((1, name.replace(FIXTURE_TAG, '')))
            else:
                result.append((type_, name))
        return result

    transformations = [transform_name] + list(sympy.parsing.sympy_parser.standard_transformations) + [untransform_name]
    ast = sympy.parsing.sympy_parser.parse_expr(string, transformations=transformations)
    return ast

def get_expression_from_string(string, signals, vectoring_dict, name,
                               parameters=None, param_suffix='',
                               bounds_dict={}):
    # tries to parse the string as an expression using sympy
    # Tokens that match signal names are mapped to that signal,
    # tokens that are unrecognized are assumed new parameters
    # Parameters named like c123 will have their names ignored; any
    # other name will be kept by creating a ConstExpression (TODO)
    '''
    Turn the string into an expression using sympy. Parameters with names that
    are not like c123 will be broken out into child expressions so the names
    appear in the user's output.
    Parameters:
        string (str): the string you want to turn into an Expression
        signals (SignalManager): existing signals, used to check which things
            in string are signals and which are coefficients, also to check
            whether things are optional signals that should be centered
        vectoring_dict (dict): Any time one of the dict keys appears in the
            expression, vector it into the values
        name (str): name for the Expression
        parameters (list): Names expected for the parameters. If None, assume
            anything unrecognized is a parameter; if not None then error when
            something is unexpected and not in the list.
        param_suffix (str): if not None, rename all parameters to append this
            suffix
        bounds_dict (dict): keep track of bounds for a particular parameter.
            Apply param_suffix and then store the bounds on the Expression.
            For now, ignore bounds that apply to vectored input parameters.
    '''
    ast = get_ast_from_string(string)


    io_symbols = {}
    param_symbols = []
    # ast.free_symbols is a set, so the iteration order seems to change
    # randomly between runs, which is annoying for debugging
    # To fix that, we explicitly sort by name here
    for sym in sorted(ast.free_symbols, key=lambda sym: sym.name):
        try:
            s = signals.from_circuit_name(sym.name)
            io_symbols[s] = sym
            continue
        except KeyError:
            # not a circuit name
            pass
        try:
            s = signals.from_template_name(sym.name)
            io_symbols[s] = sym
            continue
        except KeyError:
            # not a template name
            pass

        if parameters is None or sym.name in parameters:
            param_symbols.append(sym)
            continue

        assert False, f'Issue with parameter algebra "{string}": token "{sym.name}" is not a known input, output, analysis result, or parameter'

    def is_named(param_name):
        match = re.match('c[0-9]+', param_name)
        return not match

    # let's not do any recognition at first
    #print('param_symbols: ', param_symbols)

    # add param suffix
    # TODO think about what naming vs. not naming params means for this
    param_symbols_named = []
    bounds_named = {}
    for param_sym in param_symbols:
        new_sym = Symbol(param_sym.name + param_suffix)
        ast = ast.subs(param_sym, new_sym)
        param_symbols_named.append(new_sym)
        if param_sym.name in bounds_dict:
            bounds_named[new_sym] = bounds_dict[param_sym.name]

    # let's keep all the symbol names for now
    parent_expr = SympyExpression(ast, io_symbols, param_symbols_named,
                                  f'{name}_combiner', bounds_dict=bounds_named)

    # do vectoring - see if any of the entries in vectoring_dict are relevant
    # to this expression, and if they are then apply them
    for before, after in vectoring_dict.items():
        if before not in io_symbols:
            continue
        parent_expr.vector(before, after)

    # Create child expressions anywhere we want the user to see named things
    # First, any named parameters
    child_expressions = {}
    for param_symbol in param_symbols_named:
        param_expr = ConstExpression(param_symbol.name)
        child_expressions[param_symbol] = param_expr

    #assert len(child_expressions) == len(parent_expr.coefs)

    # Second, any optional inputs that need to be centered/broken into bits
    # we need to detect parent_expr.input_signals that should actually be
    # their own child expressions and replace them
    optional_inputs = signals.optional_expr()
    optional_input_substitutions = {}
    for input_signal in parent_expr.input_signals:
        if input_signal in optional_inputs:
            if input_signal in optional_input_substitutions:
                # already got this one
                continue
            qualifier = input_signal.friendly_name()+param_suffix
            ast, s_syms, c_syms = get_ast_from_signal(input_signal, qualifier)
            if ast == Symbol(input_signal.friendly_name()):
                # No need to create a name for this
                continue

            if isinstance(input_signal, SignalArray):
                suffix = '_bus'
                io_syms = {b: sym for b, sym in zip(list(input_signal), s_syms)}
            elif isinstance(input_signal, SignalIn):
                suffix = '_centered'
                #assert len(s_syms) == 1
                #io_syms = {input_signal: s_syms[0]}
            sig_expr = SympyExpression(ast, s_syms, c_syms,
                                       input_signal.friendly_name()+suffix)
            optional_input_substitutions[input_signal] = sig_expr

    for input_signal, expr in optional_input_substitutions.items():
        old_sym = Symbol(input_signal.friendly_name())
        new_sym = Symbol(expr.name)
        parent_expr.ast = parent_expr.ast.subs(old_sym, new_sym)
        parent_expr.input_signals.remove(input_signal)
        parent_expr.io_symbols.pop(input_signal)
        parent_expr.coefs.append(new_sym)
        parent_expr.NUM_COEFFICIENTS += 1
        child_expressions[new_sym] = expr
    parent_expr.recompile()

    #child_dict = {coef: ce for ce, coef in zip(child_expressions, parent_expr.coefs)}
    expr = HierarchicalExpression(parent_expr, child_expressions, name,
                                  bounds_dict=bounds_named)

    return expr

def get_centered_expression(expr):
    # return a new expression that is like expr, but centered
    # (or just return expr if it's already centered)
    # If expr has an offset already, replace that with a child expression that
    # always equals the right thing to center it
    # If expr has no offset, create one
    assert expr.nominal is not None, 'Cannot center non-optional expression'
    if expr.nominal == 0:
        # we are good to go
        return expr

    if expr.offset is None:
        nom = SympyExpression(expr.nominal, {},
                              list(expr.nominal.free_symbols),
                              f'{expr.name}_centering')
        nom_sym = Symbol(nom.name)
        if isinstance(expr, HierarchicalExpression):
            # edit parent expression
            # sadly we have to make a new heirarchical so it can calculate ast
            parent = SympyExpression(expr.parent_expression.ast - nom_sym,
                                     expr.parent_expression.io_symbols,
                                     expr.parent_expression.coefs + [nom_sym],
                                     expr.parent_expression.name)
            child_dict = expr.child_expressions.copy()
            child_dict[nom_sym] = nom
            expr_centered = HierarchicalExpression(parent,
                                                   child_dict,
                                                   expr.name)
            return expr_centered
        else:
            assert False, 'TODO'
        #ans.ast = ans.ast -
    else:
        assert False, 'TODO'
        #non_offset = expr.ast - expr.offset
        desired_offset = expr.nominal - expr.offset
        assert expr.offset not in desired_offset.free_symbols, 'Unexpected result'
        if expr.offset in desired_offset.free_symbols:
            # that doesn't seem right ... maybe just needs simplification?
            assert expr.offset not in simplify(desired_offset.free_symbols)
        assert False, 'TODO'


def get_optional_expression_from_influences(influence_list, name, all_signals):
    '''
    If s_list is empty, this is a constant.
    If s_list is not empty, it's a sum of expressions for each signal
    We need all_signals so we can tell what is an input when s is a string
    '''
    assert isinstance(name, str)
    def clean_expression_for_name(expression):
        return expression.replace('/', '_over_')
    individual = [ConstExpression(f'{name}_nominal')]

    # used for determining how many sample points to collect
    coefficient_counts = Counter()

    for s_or_expr in influence_list:
        if isinstance(s_or_expr, (SignalIn, SignalArray)):
            child_name = f'{name}_{s_or_expr.friendly_name()}'
            #individual.append(get_optional_expression_from_signal(s_or_expr, child_name))
            individual.append(get_default_expr_from_signal(s_or_expr, child_name))

        elif isinstance(s_or_expr, str):
            influence_name = f'{name}_<{clean_expression_for_name(s_or_expr)}>'
            expr = get_expression_from_string(s_or_expr, all_signals, {},
                                              influence_name,
                                              param_suffix=f'_{name}')
            expr_centered = get_centered_expression(expr)
            assert expr_centered.nominal == 0
            individual.append(expr_centered)
        else:
            assert False, f'Unrecognized optional expression type {type(s_or_expr)} for "{s_or_expr}"'

        for s in individual[-1].input_signals:
            coefficient_counts[s] += len(individual[-1].coefs)

    combiner = SumExpression(len(individual), f'{name}_summer')
    child_dict = {coef: ce for ce, coef in zip(individual, combiner.coefs)}
    total = HierarchicalExpression(combiner, child_dict, name)

    # for coefficient_counts, condense buses and add 1 to represent constant
    for s in coefficient_counts:
        coefficient_counts[s] += 1
    final_counts = {}
    for s, count in coefficient_counts.items():
        if not isinstance(s, SignalArray) and s.bus_info is not None:
            # this is an individual bit
            bus = all_signals.from_circuit_name(s.bus_info.bus_name)
            final_counts[bus] = max(final_counts.get(bus, 0), count)
        else:
            final_counts[s] = count
    total.optional_input_coef_counts = final_counts

    return total



def test_optimizers(error_fun, N_COEFS):
    guesses = []

    # zeros
    x0 = np.zeros(N_COEFS)
    x0[6] = 1
    x0[14] = 1
    x0[22] = 1
    guesses.append(x0)

    # ones
    x0 = np.ones(N_COEFS)
    guesses.append(x0)

    # good
    x0 = np.zeros(N_COEFS)
    for i in range(3):
        # bottom of fraction
        for j in range(7):
            x0[8*i+j] = 1e-3
        # constant offset
        x0[8*i+7] = 0
    guesses.append(x0)

    # great
    # essentially null out cm and const terms
    x0 = np.zeros(N_COEFS)
    x0[14] = 1e6
    x0[22] = 1e6
    for i in range(6):
       resistance = 2**i * 1e3
       x0[i] = 1/resistance
    # last one should be zero, but I'm afraid of divide by zero issues
    x0[6] = 1e-7
    guesses.append(x0)


    import time
    for i, x0 in enumerate(guesses):
        print('round', i, 'guess', x0)
        start = time.time()

        result = minimize(error_fun, x0, method='Nelder-Mead', options={
            #'gtol': 0,
            #'ftol': 0, # Powell
            'fatol': 0, # Nelder-Mead
            'maxfev': 5*10 ** 3,
            'disp': True
        })
        x_opt = result.x

        print('minimization took', time.time() - start)
        print('result', x_opt)
        print('nfev', result.nfev, 'success', result.success)
        e = error_fun(x_opt)
        print('error', e)
        print()
    print()





def test_vectoring():
    #coefs = [Symbol(a) for a in 'abcdef']
    #inputs = {x: Symbol(x) for x in 'uvwxyz'}
    coef_names = 'abcdef'
    input_names = 'uvwxyz'

    tests = [
        #('a*x + b', 'a0*x0 + a1*x1 + b'),
        #('b*atan(a/b*(x-c)) + d', 'b*atan(1/b*(a0*x0 + a1*x1 + a_times_c)) + d'),
        ('a*(x+b)', 'a_x0*x0 + a_x1*x1 + a_times_b'),
        # this next one is questionable because we have increased the number of
        # coefficients by splitting the vectored a and the non-vectored a**3
        # But honestly I'm not going to worry about it
        ('a*(x+b) + a**3', 'a_x0*x0 + a_x1*x1 + a_times_b + a**3'),
        ('a*c*(x+b) + y + a**3', 'c*(a_x0*x0 + a_x1*x1 + a_times_b) + a**3 + y'),
        # I think this next one is bad: we would like to distribute the a inside
        # and then distribute to the b in both terms, but instead it will
        # go to the b in the first term and the c in the second.
        # but maybe it's not a concern since you won't often have a c*b
        # ('a*((in + b) + (c*b))

    ]

    class FakeSignal(Signal):
        def __init__(self, name):
            self.name = name
            self.nominal = None
        def friendly_name(self):
            return self.name

    for before, after in tests:
        ast = get_ast_from_string(before)
        coefs = [fs for fs in ast.free_symbols if fs.name in coef_names]
        inputs = {FakeSignal(fs.name): fs
                  for fs in ast.free_symbols if fs.name in input_names}
        x = [s for s in inputs if s.name == 'x'][0]


        se = SympyExpression(ast, inputs, coefs, 'vec_test_expr')
        se.vector(x, [FakeSignal('x0'), FakeSignal('x1')])
        after_ast = get_ast_from_string(after)
        assert se.ast == after_ast

if __name__ == '__main__':
    test_vectoring()