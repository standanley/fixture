import re
from abc import ABC, abstractmethod
from collections import defaultdict
from keyword import iskeyword

import sympy
from scipy.optimize import minimize, basinhopping, _basinhopping
from scipy.stats import linregress
import numpy as np
import pandas

#import fixture
from sympy import Symbol, lambdify, Mul, Add

import fixture.regression
from fixture.fitting_tricks import init_tricks
from fixture.sampler import SampleManager
from fixture.signals import SignalArray, SignalIn, CenteredSignalIn, Signal
from fixture.plot_helper import plt, PlotHelper

PLOT = True

class Expression(ABC):
    # TODO I would like for these to be abstract properties, but also they are
    #  not required until the end of __init__, so I don't think abc can do that
    input_signals = None
    NUM_COEFFICIENTS = None

    x_opt = None
    last_coef_is_offset = False

    def __init__(self, name):
        # TODO when I override this, instead of calling super(), I've just been
        # rewriting this one line to set the name. Is that bad practice? probably
        self.name = name

    #@property
    #@abstractmethod
    #def NUM_COEFFICIENTS(self):
    #    # a constant property that is the number of coefs to fit
    #    # Remember that the constant offset is separate from this count
    #    return self._NUM_COEFFICIENTS
    #@NUM_COEFFICIENTS.setter
    #def NUM_COEFFICIENTS(self, value):
    #    # TODO see note on input_signals
    #    self._NUM_COEFFICIENTS = value


    #@property
    #@abstractmethod
    #def input_signals(self):
    #    # a list of signals whose values this expression depends on. Those
    #    # values will be passed, in the same order, to predict() and fit()
    #    return self._input_signals
    #@input_signals.setter
    #def input_signals(self, value):
    #    # TODO this is to allow the value to be set in __init__ ...
    #    # There may be a better way to implement these abstract properties
    #    self._input_signals = value

    x_init = None

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

        # check init_tricks
        io_syms = list(self.io_symbols.values())
        coef_syms = self.coefs
        #io_reverse = {sym: sig for sig, sym in io_syms.items()}
        data_by_input_sym = {self.io_symbols[sig]: my_data[sig] for sig in self.input_signals}
        data_by_input_sym = pandas.DataFrame(data_by_input_sym)
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
        print('Minimizing', self.name)
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
        displace = _basinhopping.RandomDisplacement(stepsize=1.0)
        take_step_wrapped = _basinhopping.AdaptiveStepsize(displace,
                                            interval=3,
                                            factor=0.1,
                                            verbose=False)
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

        result = basinhopping(
            error,
            x0,
            niter=10**1,
            T=temperature,
            #take_step=take_step_wrapped,
            take_step=random_step,
            minimizer_kwargs={
                'method':'Nelder-Mead',
                'options' : {
                    #'ftol': 1e-8, # Powell
                    #'fatol': 0, # Nelder-Mead
                    'maxfev': 10**3,
                    'return_all': False
                }
            }
        )
        x_opt = result.x

        print('minimization took', time.time() - start, 'error', error(x_opt))
        self.x_opt = x_opt
        print(x0)
        print(x_opt)
        print()


        if False:
            from fixture.plot_helper import plt

            # TODO this is a temporary plot
            predict_vec = np.vectorize(self.predict, signature='(n),(m)->()')
            predictions = predict_vec(my_data, x0)
            x_label = my_data.columns[0]
            plt.title('TEMP looking at x_init')
            plt.scatter(my_data[x_label], result_data)
            plt.scatter(my_data[x_label], predictions)
            plt.grid()
            plt.show()

            predict_vec = np.vectorize(self.predict, signature='(n),(m)->()')
            predictions = predict_vec(my_data, x_opt)
            x_label = my_data.columns[0]
            plt.scatter(my_data[x_label], result_data)
            plt.scatter(my_data[x_label], predictions)
            plt.grid()
            plt.show()


        return x_opt

    def eval(self, data):
        # for use after fitting has been done
        assert isinstance(data, (dict, pandas.DataFrame))
        assert self.x_opt is not None
        print()
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
    def __init__(self, ast, io_symbols, coefs, name):
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

        self.recompile_fun()

    def recompile_fun(self):
        input_syms = [self.io_symbols[s] for s in self.input_signals]
        self.fun_compiled = lambdify([input_syms, self.coefs], self.ast, cse=True)
        #test = self.fun_compiled([1, 2], [3, 4, 5])
        #print(test)


    def predict(self, opt_values, coefs):
        #input_syms = [self.io_symbols[s] for s in self.input_signals]
        ##fun = lambdify(input_syms+self.coefs, self.ast, cse=True)
        #args_sym = input_syms + self.coefs
        #args_val = list(opt_values) + list(coefs)
        ##ans = fun(*args)
        #ans = self.ast.subs(zip(args_sym, args_val)).evalf()

        # lambdify should be smart enough to accept the input in list form
        # or not
        ans = self.fun_compiled(opt_values, coefs)
        return ans

    def predict_many(self, opt_value_data, coefs):
        #input_syms = [self.io_symbols[s] for s in self.input_signals]
        #substitutions = zip(self.coefs, coefs)
        #ast_coefs = self.ast.subs(substitutions)
        #fun = lambdify([input_syms], ast_coefs, cse=True)#, docstring_limit=-1)
        #data = opt_value_data[self.input_signals].values
        #ans = fun(data.T)
        #return ans
        #data = opt_value_data[self.input_signals].values
        #data_all = np.concatenate((data, np.array(coefs)))


        ans = self.fun_compiled(opt_value_data.values.T, coefs)

        #ans = np.array([self.predict(opt_row, coefs) for opt_row in opt_value_data.values])
        return ans


    def verilog(self, lhs, coef_names):
        assert isinstance(lhs, str)
        print('TODO: SympyExpression verilog')
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

        self.recompile_fun()
        return


    def vector_fallback(self, signal, new_signals, signal_sym, new_signals_sym):
        '''
        Vector the signal by replacing it with a dot product of its
        components and new weight coefficients.
        self.vector() will do something better and try to absorb an existing
        scaling coefficient into the weights, but if that fails it will call
        this function as a fallback.
        '''
        new_coefs_sym = [Symbol(f'{s.friendly_name()}_weight_{self.name}')
                       for s in new_signals]

        sym_vectored = Add(Mul(c, s)
                           for c, s in zip(new_coefs_sym, new_signals_sym))
        self.ast.subs((signal_sym, sym_vectored))
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

        # search for all the instances of signal in the ast, noting specially
        # the places where it appears in a  product
        products = []
        appears_outside_product = False
        def rec(ast):
            if isinstance(ast, Symbol):
                if ast == signal_sym:
                    appears_outside_product = True
                return
            if ast.is_Mul and signal_sym in ast.args:
                products.append(ast)
                return
            for arg in ast.args:
                rec(arg)
        rec(self.ast)

        if appears_outside_product:
            self.vector_fallback()
            return
        assert len(products) > 0, 'Internal error in equation vectoring'

        coefs = []
        for product in products:
            this_product_coefs = []
            for factor in product.args:
                if factor == signal_sym:
                    continue
                elif isinstance(factor, Symbol):
                    this_product_coefs.append(factor)
                else:
                    # this factor is a whole ast; not what we are looking for
                    continue
            coefs.append(this_product_coefs)
        assert len(coefs) > 0, 'Internal error in equation vectoring'

        # For each appearance of signal in ast there is an entry in coefs.
        # Each entry in coefs is a list of the symbols that are multiplied by
        # the signal.
        # Our goal is to find a symbol that is multiplied by the signal every
        # time it appears.
        results = []
        for candidate in coefs[0]:
            if all(candidate in cs for cs in coefs):
                results.append(candidate)

        if len(results) == 0:
            # not able to find a coef for signal; fall back to vectoring it
            # in-place
            self.vector_fallback()
            return

        coef_to_vector = results[0]
        if len(results) > 1:
            print(f'Unexpected case: in parameter equation {self.ast}, signal {signal} has multiple coefficients {results}, and Fixture does not know which one to vector, but is choosing {coef_to_vector}')

        new_coefs = [Symbol(f'{coef_to_vector.name}_{s.friendly_name()}') for s in new_signals]
        new_expr = Add(*[Mul(c, s) for c, s in zip(new_coefs, new_signals_sym)])
        self.ast = self.ast.subs([(Mul(coef_to_vector, signal_sym), new_expr)])

        if coef_to_vector in self.ast.free_symbols:
            print(f'Warning: when vectoring, the coefficient {coef_to_vector} was replaced by {new_coefs}, but some instances of {coef_to_vector} still remain in the expression "{self.ast}"')
            old_coefs = []
        else:
            old_coefs = [coef_to_vector]

        self.vector_cleanup(signal, new_signals, new_signals_sym,
                            old_coefs, new_coefs)


    def copy(self, param_suffix):
        assert False

class AnalogExpression(SympyExpression):
    NUM_COEFFICIENTS = 2
    last_coef_is_offset = True

    def __init__(self, opt_signal, name, centering_offset=0):
        #self.name = name
        #self.input_signals = [opt_signal]
        io_symbols = {opt_signal: Symbol(opt_signal.friendly_name())}
        coefs = [Symbol(f'gain_{name}'), Symbol(f'offset_{name}')]
        ast = coefs[0]*(io_symbols[opt_signal]-centering_offset) + coefs[1]
        super().__init__(ast, io_symbols, coefs, name)

    #def predict(self, opt_values, coefficients):
    #    assert len(opt_values) == len(self.input_signals)
    #    assert len(coefficients) == 2
    #    return opt_values[0] * coefficients[0] + coefficients[1]

    #def verilog(self, lhs, coef_names):
    #    opt_names = [self.friendly_name(s) for s in self.input_signals]
    #    assert len(opt_names) == 1
    #    assert len(coef_names) == 2
    #    return [f'{lhs} = {coef_names[0]}*{opt_names[0]} + {coef_names[1]};']

class ConstExpression(Expression):
    NUM_COEFFICIENTS = 1
    input_signals = []
    last_coef_is_offset = True

    def predict(self, opt_values, coefficients):
        assert len(opt_values) == 0
        assert len(coefficients) == 1
        return coefficients[0]

    def verilog(self, lhs, coef_names):
        opt_names = [self.friendly_name(s) for s in self.input_signals]

        assert len(opt_names) == 0
        assert len(coef_names) == 1
        return [f'{lhs} = {coef_names[0]};']


class AffineExpression(Expression):
    __doc__ = '''The constructor takes in a list of inputs. A coefficient will 
                 be assigned to each one. There is a constant at the end. '''
    last_coef_is_offset = True

    def __init__(self, inputs, name):
        assert False, 'AffineExpression is still a work in progress'
        self.name = name
        self.input_signals = inputs
        self.NUM_COEFFICIENTS = len(inputs) + 1

    def predict(self, opt_values, coefs):
        # opt_values are essentially input_values
        assert len(opt_values) == len(self.input_signals)
        assert len(coefs) == len(opt_values)
        return sum(o*c for o, c in zip(opt_values, coefs[:-1])) + coefs[-1]

    def fit(self, optional_data, result_data):
        print('using linear regression')
        # we want to get a rough fit before we go to the solver
        # we can guess that the bits are thermometer or binary up/down
        # If we know the ratios between bits, we can do linear regression
        opt_values = np.array([optional_data[s] for s in self.input_signals])
        result = np.linalg.pinv(opt_values.T) @ result_data
        self.x_opt = result
        assert False, 'offset?'
        return self.x_opt

    def verilog(self, lhs, coef_names):
        opt_names = [self.friendly_name(s) for s in self.input_signals]
        assert False, 'todo'

class LinearExpression(Expression):
    __doc__ = '''The constructor takes in a list of inputs. A coefficient will 
                 be assigned to each one. There is NO constant at the end. '''
    last_coef_is_offset = False

    def __init__(self, inputs, name):
        self.name = name
        self.input_signals = inputs
        # coefficients do NOT include any constant offset, because often the
        # last input is already a constant
        self.NUM_COEFFICIENTS = len(inputs)

    def predict(self, opt_values, coefs):
        # opt_values are essentially input_values
        assert len(opt_values) == len(self.input_signals)
        assert len(coefs) == len(opt_values)
        return sum(o*c for o, c in zip(opt_values, coefs))

    def fit(self, optional_data, result_data):
        # we want to get a rough fit before we go to the solver
        # we can guess that the bits are thermometer or binary up/down
        # If we know the ratios between bits, we can do linear regression
        opt_values = np.array([optional_data[s] for s in self.input_signals])
        result = np.linalg.pinv(opt_values.T) @ result_data
        self.x_opt = result
        return self.x_opt

    def verilog(self, lhs, coef_names):
        opt_names = [self.friendly_name(s) for s in self.input_signals]
        assert len(opt_names) == self.NUM_COEFFICIENTS
        assert len(coef_names) == self.NUM_COEFFICIENTS
        ans = ' + '.join(f'{cn}*{on}' for on, cn in zip(opt_names, coef_names))
        return [f'{lhs} = {ans};']

class HeirarchicalExpression(Expression):
    def __init__(self, parent_expression, child_expressions, name):
        # each coefficient in the parent expression is the result of evaluating
        # a child expression
        # child_expression_names should match the parent inputs
        self.name = name
        self.manage_offsets = isinstance(parent_expression, SumExpression)


        # TODO I think there could be a bug if we are managing coefficients
        #  but one of the child expressions is a ConstExpression
        assert parent_expression.NUM_COEFFICIENTS == (len(child_expressions) + (1 if self.manage_offsets else 0))
        self.parent_expression = parent_expression
        self.child_expressions = child_expressions

        child_inputs = [s for ce in child_expressions for s in ce.input_signals]
        self.input_signals = list(parent_expression.input_signals) + child_inputs

        num_child_coefficients = sum(ce.NUM_COEFFICIENTS for ce in child_expressions)
        # don't add num parent coefficients because those are the child expressions
        if self.manage_offsets:
            # we do something special here by consolidating child offsets
            num_children_offsets = 0
            for child in self.child_expressions:
                if child.last_coef_is_offset:
                    num_children_offsets += 1
            self.num_redundant_offsets = num_children_offsets-1 if num_children_offsets > 0 else 0
            self.NUM_COEFFICIENTS = num_child_coefficients - self.num_redundant_offsets
        else:
            self.num_redundant_offsets = 0
            self.NUM_COEFFICIENTS = num_child_coefficients

        # compile our own ast

        io_symbols = self.parent_expression.io_symbols.copy()
        coefs = []
        assert isinstance(parent_expression, SympyExpression)
        if self.manage_offsets:
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
            child_asts = []
            for sym, child in zip(self.parent_expression.coefs, self.child_expressions):
                if isinstance(child, ConstExpression):
                    child_asts.append(sym)
                    coefs.append(sym)
                    continue
                child_asts.append(child.ast)
                io_symbols.update(child.io_symbols)
                coefs += child.coefs

        final_ast = self.parent_expression.ast
        for sym, child_ast in zip(self.parent_expression.coefs, child_asts):
            final_ast = final_ast.subs(sym, child_ast)

        if self.manage_offsets:
            # TODO I don't think I specified in the definition of the parent
            #  expression that the offset_sym has to be the last coef
            # parent_expression already has extra offset, just need it in coefs
            offset_sym = parent_expression.coefs[-1]
            coefs.append(offset_sym)

        self.ast = final_ast
        self.io_symbols = io_symbols
        self.coefs = coefs
        assert len(self.coefs) == self.NUM_COEFFICIENTS

        # self.inputs probably has repeats,
        # but self.io_symbols.keys() should be the same set without repeats
        assert set(self.input_signals) == set(self.io_symbols)
        self.input_signals = list(self.io_symbols.keys())
        input_syms = [self.io_symbols[s] for s in self.input_signals]

        for sym in self.ast.free_symbols:
            assert sym in input_syms or sym in self.coefs

        self.fun_compiled = lambdify([input_syms, self.coefs], self.ast, cse=True)








    def fit_by_group(self, optional_data, result_data):
        # relies on a specific structure of heirarchy and specific sweep groups
        # in optional_data in order to fit children and grandchildren one
        # piece at a time
        # TODO I'm still getting nan instead of None for sg sometimes
        groups = {sg for sg in optional_data[SampleManager.GROUP_ID_TAG]
                  if (sg is not None and not isinstance(sg, float))}
        groups = sorted(groups, key=lambda sg: sg.name)

        # first thing is to figure out which sweep groups to use for which
        # fit expressions
        fits_for_sweeps = defaultdict(list)
        for i in range(len(self.child_expressions)):
            child = self.child_expressions[i]
            for grandchild in child.child_expressions:
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

                self.parent_expression.fit(point_data, point_data_res,
                                           parent_by_group_x_init)
                result = self.parent_expression.x_opt
                error = self.parent_expression.error(
                    point_data[self.parent_expression.input_signals],
                    self.parent_expression.x_opt,
                    point_data_res)
                return result, error


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
            for si in sorted(sweep_ids):
                point_data, point_data_res, example_row_id = get_data_for_id(si)
                self.parent_expression.fit(point_data, point_data_res,
                                           parent_by_group_x_init)
                parent_by_group_x_init = self.parent_expression.x_opt
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
                    for xss in plot_xss:
                        vals = []
                        for s in sg.signals:
                            v = list(xss[s])[0]
                            v_str = str(v) if isinstance(v, int) else f'{v:.4g}'
                            vals.append(f'{s.friendly_name()}={v_str}')
                        legend.append(', '.join(vals))

                    plt.legend(legend)
                    plt.title(f'Fitting {self.name} for various {sg}')
                    plt.xlabel(f'{xaxis.friendly_name()}')
                    plt.ylabel(f'{self.parent_expression.name}')
                    plt.grid()
                    PlotHelper.save_current_plot(f'individual_fits/{self.name}/{sg.name}/Fits for {self.name} vs {xaxis.friendly_name()} from Sweeping {sg.name}')

                    # now plot the worst one by itself
                    plt.figure()
                    worst_i = sorted(sweep_ids).index(worst_example[0])
                    xss = plot_xss[worst_i]
                    xs = xss[xaxis]
                    ys = plot_ys[worst_i]
                    preds = plot_predictions[worst_i]

                    order = np.argsort(xs)
                    plt.plot(np.array(xs)[order], np.array(ys)[order], '*')
                    plt.plot(np.array(xs)[order], np.array(preds)[order], '--')
                    plt.legend('measured', 'predicted')
                    plt.title(f'Fitting {self.name}, worst fit at {sg}=TODO')
                    plt.grid()
                    PlotHelper.save_current_plot(f'individual_fits/{self.name}/{sg.name}/Worst fit for {self.name} vs {xaxis.friendly_name()} from Sweeping {sg.name}')


            # Next we find expressions for each param, using values from earlier
            # each row in example_data corresponds to one point in the sweep
            # TODO I'd like to delete the columns corresponding to test inputs
            #  in example_data because they are not relevant. But I don't have
            #  a good way to find the names for those. It's just confusing for
            #  somebody maintaining the code
            example_data = group_data.loc[example_row_ids]
            for i in range(len(self.child_expressions)):
                child = self.child_expressions[i]
                child_results = group_results[:, i]

                # When we fit by group we don't exactly follow the normal
                # hierarchy for the child, so it has to be in a specific form
                # TODO move this assertion to the top?
                assert isinstance(child, HeirarchicalExpression)
                assert isinstance(child.parent_expression, SumExpression)

                ## now we look through child's children and determine which
                # one(s) are actually affected by changing sg
                # TODO sg.signal doesn't always exist?
                # TODO what about when sg has multiple signals?
                # TODO what about when child.input_signals has multiple signals?
                #relevant_grandchildren = []
                for grandchild in child.child_expressions:
                    if grandchild not in fits_for_sweeps[sg]:
                        continue
                    # we are here because this grandchild actually depends on
                    # the sg we are currently working with
                    grandchild.fit(example_data, child_results)

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
                        predictions = grandchild.predict_from_dict(data_smooth, grandchild.x_opt)
                        xs_smooth = data_smooth[xaxis]

                        xs_order = np.argsort(xs)
                        plt.plot(np.array(xs)[xs_order], child_results[xs_order], '*')
                        plt.plot(xs_smooth, predictions, 'x--')
                        plt.xlabel(f'{xaxis.friendly_name()}')
                        plt.ylabel(f'{child.name}')
                        plt.grid()
                        PlotHelper.save_current_plot(f'individual_fits/{self.name}/{sg.name}/Individual fit for {grandchild.name} vs {sg}')

                        # TEMP for checking x_init
                        if grandchild.x_init is not None:
                            plt.figure()
                            predictions = grandchild.predict_from_dict(data_smooth, grandchild.x_init)
                            #plt.plot(xs, child_results, '*')
                            plt.plot(np.array(xs)[xs_order],
                                     child_results[xs_order], '*')
                            plt.plot(xs_smooth, predictions, 'x--')
                            plt.xlabel(f'{xaxis.friendly_name()}')
                            plt.ylabel(f'{child.name}')
                            plt.grid()
                            PlotHelper.save_current_plot(f'individual_fits/{self.name}/{sg.name}/debug/Initial point for minimizer for {grandchild.name} vs {sg}')
                        print()


                    result_fits[grandchild].append(grandchild.x_opt)

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

        # now we are ready
        self.fit(optional_data, result_data)

        if PLOT:
            #predict_data = [optional_data[col] for col in self.input_signals]
            predict_data = optional_data[self.input_signals]
            predictions = self.predict_many(predict_data, self.x_opt)
            # TODO fix this xaxis
            xaxis = self.parent_expression.input_signals[0]
            xs = optional_data[xaxis]
            plt.figure()
            plt.plot(xs, result_data, '*')
            plt.plot(xs, predictions, '--')
            plt.grid()
            plt.xlabel(f'{xaxis.friendly_name()}')
            plt.ylabel(f'{self.name}')
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

            # take the next slice of coefs for ce, but if ce has a redundant
            # offset then we should just pass it a zero for that offset instead
            # num_child_coefficients is the number of our coefficients to slice
            # out for this child, so it does NOT include the offset
            redundant_offset = self.manage_offsets and ce.last_coef_is_offset
            num_child_coefficients = ce.NUM_COEFFICIENTS
            if redundant_offset:
                num_child_coefficients -= 1
            coefs_ce = coefs[coef_count : coef_count + num_child_coefficients]
            coef_count += num_child_coefficients
            if redundant_offset:
                coefs_ce = np.concatenate((coefs_ce, [0]))

            parent_coef = ce.predict(input_values_ce, coefs_ce)
            parent_coefs.append(parent_coef)

        if self.manage_offsets:
            # should be 1 left over for the aggregate offset
            parent_coefs.append(coefs[coef_count])
            coef_count += 1

        assert coef_count == len(coefs)

        parent_input_indices = [self.input_signals.index(s) for s in
                            self.parent_expression.input_signals]
        parent_input_values = [opt_values[i] for i in parent_input_indices]
        ans = self.parent_expression.predict(parent_input_values, parent_coefs)
        return ans

    def predict_from_dict(self, opt_dict, coefs):
        assert False
        return None

    def predict_many(self, opt_value_data, coefs):
        ans = self.fun_compiled(opt_value_data.values.T, coefs)
        return ans


    @property
    def x_opt(self):
        return self._x_opt
    @x_opt.setter
    def x_opt(self, x_opt):
        # set self.opt for self and all the children
        # the incoming x_opt is short because it has only the aggregate offset
        # the final x_opt_new is long because is has the offset for each child
        # and also the aggregate offset (shifted to account for the child
        # offsets)
        managed_offset_acc = 0
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
                managed_offset_acc += offset_nom
                x_opt_ce = np.concatenate((x_opt_ce, [offset_nom]))
                x_opt_new = np.insert(x_opt_new, x_opt_new_count, offset_nom)
                x_opt_new_count += 1
            ce.x_opt = x_opt_ce

        if self.manage_offsets:
            # managed offset x_opt[-1] doesn't go into any ce.x_opt
            # but it does get edited to account for offsets already given to ces
            x_opt_new[-1] -= managed_offset_acc
            x_opt_count += 1
            x_opt_new_count += 1

        assert x_opt_count == len(x_opt)
        assert x_opt_new_count == len(x_opt_new)

        # we do this at the end because it can get edited with managed offsets
        self._x_opt = x_opt_new

    @property
    def x_init(self):
        # we need to aggregate x_init from children, taking care with
        # redundant offsets. Returns a single aggregate offset, assuming the
        # children are each passed zero for their redundant offsets
        x_init = []
        # aggregate offset is a correction for the individual offsets being 0
        # nominal_offsets is our guess for the actual nominal offset
        aggregate_offset = 0
        nominal_offsets = []
        for ce in self.child_expressions:
            ce_x_init = ce.x_init if ce.x_init is not None else np.ones(ce.NUM_COEFFICIENTS)
            if self.manage_offsets and ce.last_coef_is_offset:
                x_init = np.concatenate((x_init, ce_x_init[:-1]))

                # when we shrink x_init to have only 1 offset, no redundant ones,
                # we assume that children will be passed 0 for their redundant
                # offsets
                ce_inputs_nom = [s.nominal for s in ce.input_signals]
                nominal_offsets.append(ce.predict(ce_inputs_nom, ce_x_init))
                aggregate_offset += nominal_offsets[-1] - ce_x_init[-1]
            else:
                x_init = np.concatenate((x_init, ce_x_init))

        if self.manage_offsets:
            nominal_offset = sum(nominal_offsets) / len(nominal_offsets)
            final_offset = nominal_offset - aggregate_offset
            x_init = np.concatenate((x_init, [final_offset]))

        return x_init
    @x_init.setter
    def x_init(self, x_init):
        raise NotImplementedError('Cannot set heirarchical x_init directly')

    def _search(self, target_name):
        # return a list of child expressions with the target name
        matches = []
        if self.name == target_name:
            matches.append(self)

        for child in self.child_expressions:
            if isinstance(child, HeirarchicalExpression):
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
        def name(expr):
            # seems like expr.name is already qualified with self.name
            return f'{expr.name}'
        name_nominal = f'{self.name}_nominal'
        lines = []

        # all the child expressions
        coef_count = 0
        for ce in self.child_expressions:
            slice_len = ce.NUM_COEFFICIENTS
            if self.manage_offsets and ce.last_coef_is_offset:
                slice_len -= 1
            child_coefs = coef_names[coef_count : coef_count + slice_len]
            if self.manage_offsets and ce.last_coef_is_offset:
                child_coefs = list(child_coefs) + ['0']
            coef_count += slice_len

            lines += ce.verilog(name(ce), child_coefs)
        if self.manage_offsets:
            lines.append(f'{name_nominal} = {coef_names[coef_count]};')
            coef_count += 1
        assert coef_count == len(coef_names)

        # parent expression next
        parent_coef_names = [name(ce) for ce in self.child_expressions]
        if self.manage_offsets:
            parent_coef_names.append(name_nominal)
        assert len(parent_coef_names) == self.parent_expression.NUM_COEFFICIENTS
        lhs_str = lhs if isinstance(lhs, str) else lhs.friendly_name()
        lines += self.parent_expression.verilog(lhs_str, parent_coef_names)

        # TODO I want to reverse this properly so the c numbers are in order,
        #  But I'm afraid I will mess it up and they'll be mismatched
        return lines[::-1]

    #def copy(self, param_suffix):
    #    # for use with vectoring outputs, copy this whole tree but rename the
    #    # parameters with the given suffix
    #    parent = self.parent_expression.copy(param_suffix)
    #    children = [c.copy(param_suffix) for c in self.child_expressions]
    #    name = f'{self.name}_{param_suffix}'
    #    ans = HeirarchicalExpression(parent, children, name)
    #    return ans

class SumExpression(SympyExpression):
    input_signals = []

    def __init__(self, n, name):
        #self.name = name
        #self.NUM_COEFFICIENTS = n

        io_symbols = {}
        coefs = [Symbol(f'{name}_c{i}') for i in range(n)]
        ast = sum(coefs)
        super().__init__(ast, io_symbols, coefs, name)

    #def predict(self, opt_values, coefs):
    #    assert len(opt_values) == 0
    #    return sum(coefs)

    def verilog(self, lhs, coef_names):
        opt_names = [self.friendly_name(s) for s in self.input_signals]
        assert len(opt_names) == 0
        assert len(coef_names) == self.NUM_COEFFICIENTS
        ans = ' + '.join(f'{cn}' for cn in coef_names)
        return [f'{lhs} = {ans};']

#class CenterExpression(Expression):
#
#    def __init__(self, s):
#        self.input_signals = [s]
#        self.name = f'{s.friendly_name()}_deviation'
#        self.NUM_COEFFICIENTS = 0
#
#    def predict(self, opt_values, coefs):
#        assert len(opt_values) == 1
#        assert len(coefs) == 0
#        return opt_values[0] - self.input_signals[0].nominal
#
#    def verilog(self, lhs, coef_names):
#        opt_names = [self.friendly_name(s) for s in self.input_signals]
#        assert len(coef_names) == 0
#        return [f'{self.name} = {opt_names[0]} - {self.input_signals[0].nominal};']


def center_expression_inputs(ExpressionUncentered, signals_to_center):
    '''
    Given an Expression class, return a modified version of the class that
    centers the signals in signals_to_center before using them. The generated
    Verilog will include extra lines to do this centering.
    '''
    assert issubclass(ExpressionUncentered, SympyExpression)
    name_mapping = {s: f'{s.friendly_name()}_deviation'
                    for s in signals_to_center}
    ast = ExpressionUncentered

    #class ExpressionCentered(ExpressionUncentered):
    #    def __init__(self, *args, **kwargs):
    #        temp = super().__init__(*args, **kwargs)

    #        # we precompute offsets to make predict() faster
    #        offsets = [-s.nominal if s in signals_to_center else 0
    #                   for s in self.input_signals]
    #        self.centering_offsets = np.array(offsets)

    #        # it's tempting to edit self.signals here so we take in
    #        # s_deviation instead of s, but that confuses things when we are
    #        # a child in a Heirarchical thing and the parent needs to know
    #        # to pass us s rather than s_deviation
    #        return temp

    #    def predict(self, opt_values, coefs):
    #        assert len(opt_values) == len(self.input_signals)
    #        if isinstance(opt_values, list):
    #            opt_values = np.array(opt_values)
    #        return super().predict(opt_values - self.centering_offsets, coefs)

    #    def fit(self, optional_data, result_data):
    #        optional_data_centered = optional_data.copy()
    #        for s in signals_to_center:
    #            optional_data_centered[name_mapping[s]] = optional_data[s] - s.nominal
    #        return super().fit(optional_data_centered, result_data)

    #    def verilog(self, lhs, coef_names):

    #        lines = []
    #        for s in signals_to_center:
    #            lines.append(f'{name_mapping[s]} = {s.friendly_name()} - {s.nominal};')


    #        # we want to temporarily change the names of our signals so the
    #        # call to ExpressionUncentered's verilog uses s_deviation
    #        input_signals_uncentered = self.input_signals
    #        self.input_signals = [name_mapping[s] if s in signals_to_center else s
    #                              for s in input_signals_uncentered]
    #        lines += super().verilog(lhs, coef_names)
    #        self.input_signals = input_signals_uncentered

    #        return lines

    #return ExpressionCentered





class ReciprocalExpression(SympyExpression):
    __doc__ = '''
The constructor takes in a list of inputs. This is intended to model the
resistance of a bank of resistors. 
1/(ctrl[0]/R0 + ctrl[1]/R1 + ... + 1/Rnom)
Issue is that this expression is difficult to zero out for parameters that don't
depend on R. An alternative is:
Rnom/(ctrl[0]/X0 + ctrl[1]/X1 + ... + 1)
Issue with that one is that it's difficult if the crtl=0 resistance is 0.
We end up going with this weird version:
G/(ctrl[0] + ctrl[1]/X1 + ... + Y)
I still consider this a work-in-progress. I think it makes sense if you 
interpret G as a sort of "global conductance multiplier" for each bit
1/(ctrl[0]/Rnom + ctrl[1]/(Rnom*X1) + ctrl[2]/(Rnom*X2) + ... + Y/Rnom)
Computationally, we pull Rnom on the top to avoid precision weirdness
Note that Y is on the top of that last term because Y=0 is common, Y=inf is not
For ctrl[5:0], coefs = [Rnom, X1, X2, X3, X4, X5, Y]
Also, there is a additional additive offset at the end
For ctrl[5:0], coefs = [Rnom, X1, X2, X3, X4, X5, Y, offset]
'''
    last_coef_is_offset = True

    def __init__(self, inputs, name):
        #self.name = name
        #self.input_signals = inputs
        #self.NUM_COEFFICIENTS = len(inputs) + 2
        self.x_init = np.ones(self.NUM_COEFFICIENTS)

        ## TODO get rid of this
        #self.x_init[0] = 1500
        #self.x_init[1] = 2
        #self.x_init[2] = 4
        #self.x_init[3] = 8
        #self.x_init[4] = 8
        #self.x_init[5] = 8
        #self.x_init[6] = 0

        io_symbols = {s: Symbol(s.friendly_name()) for s in inputs}
        coefs = [Symbol(f'{name}_c{i}') for i in range(len(inputs)+2)]
        opt_values = [io_symbols[s] for s in inputs]
        denominator = opt_values[0] + sum(o / c for o, c in zip(opt_values[1:], coefs[1:-2])) + coefs[-2]
        ast = coefs[0] / denominator + coefs[-1]
        super().__init__(ast, io_symbols, coefs, name)


    #def predict(self, opt_values, coefs):
    #    # opt_values are essentially input_values
    #    assert len(opt_values) == len(self.input_signals)
    #    assert len(coefs) == len(opt_values) + 2
    #    denominator = opt_values[0] + sum(o / c for o, c in zip(opt_values[1:], coefs[1:-2])) + coefs[-2]
    #    return coefs[0] / denominator + coefs[-1]

    def get_x_init(self, optional_data, result_data):
        # we want to get a rough fit before we go to the solver
        # we can guess that the bits are thermometer or binary up/down
        # If we know the ratios between bits, we can do linear regression
        assert False, 'trying to phase this out, right?'

    def fit(self, optional_data, result_data):
        def quick_fit(xs, ys):
            result = linregress(xs, ys)
            return result.rvalue**2, result.slope, result.intercept
        bits = [optional_data[s] for s in self.input_signals]
        denom_coefss = []
        therm = [1 for i in range(1, len(bits))] + [0]
        denom_coefss.append(therm)
        binary_inc = [2**i for i in range(1, len(bits))] + [1e-3]
        denom_coefss.append(binary_inc)
        binary_dec = [(1/2)**i for i in range(1, len(bits))] + [1e-3]
        denom_coefss.append(binary_dec)

        # r2_value, slope, intercept, denom_coefs
        # r2_value is first so we can use comparison to find the highest
        fits = []
        for denom_coefs in denom_coefss:
            denom = sum(b/c for b, c in zip(bits, [1]+denom_coefs[:-1])) + denom_coefs[-1]
            xs = 1/denom
            result = quick_fit(xs, result_data)
            fits.append((*result, denom_coefs))
        fit = max(fits, key=lambda info: info[0])
        r2_value, slope, intercept, denom_coefs = fit

        coefs = [slope] + denom_coefs + [intercept]
        self.x_init = coefs
        return super().fit(optional_data, result_data)

    def verilog(self, lhs, coef_names):
        opt_names = [self.friendly_name(s) for s in self.input_signals]
        assert len(opt_names) == self.NUM_COEFFICIENTS - 2
        assert len(coef_names) == self.NUM_COEFFICIENTS
        bit_weights = ['1'] + list(coef_names[1:-2])
        denom = ' + '.join(f'{on}/{cn}' for on, cn in zip(opt_names, bit_weights))
        denom += f' + {coef_names[-2]}'
        ans = f'{coef_names[0]} / ({denom}) + {coef_names[-1]}'
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

def get_expression_from_string(string, signals, parameters, name, vectoring_dict, param_suffix=''):
    # tries to parse the string as an expression using sympy
    # Tokens that match signal names are mapped to that signal,
    # tokens that are unrecognized are assumed new parameters
    # Parameters named like c123 will have their names ignored; any
    # other name will be kept by creating a ConstExpression (TODO)
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

        if sym.name in parameters:
            param_symbols.append(sym)
            continue

        assert False, f'Issue with parameter algebra "{string}": token "{sym.name}" is not a known input, output, analysis result, or parameter'

    def is_named(param_name):
        match = re.match('c[0-9]+', param_name)
        return not match

    # let's not do any recognition at first
    print(param_symbols)

    # add param suffix
    # TODO think about what naming vs. not naming params means for this
    param_symbols_suffix = []
    for param_sym in param_symbols:
        new_sym = Symbol(param_sym.name + param_suffix)
        ast = ast.subs(param_sym, new_sym)
        param_symbols_suffix.append(new_sym)

    # let's keep all the symbol names for now
    parent_expr = SympyExpression(ast, io_symbols, param_symbols_suffix, f'{name}_combiner')

    # do vectoring - see if any of the entries in vectoring_dict are relevant
    # to this expression, and if they are then apply them
    for before, after in vectoring_dict.items():
        if before not in io_symbols:
            continue
        parent_expr.vector(before, after)

    child_expressions = []
    for param_symbol in param_symbols_suffix:
        param_expr = ConstExpression(param_symbol.name)
        child_expressions.append(param_expr)


    expr = HeirarchicalExpression(parent_expr, child_expressions, name)


    return expr



def get_optional_expression_from_signal(s, name):
    if isinstance(s, SignalArray):
        print('TODO fix SignalArray behavior, right now defaulting to ReciprocalExpression')
        #return LinearExpression(list(s), name)
        return ReciprocalExpression(list(s), name)

    else:
        assert isinstance(s, SignalIn)
        # TODO trying to phase out usage of s.type_
        #assert s.type_ in ['real', 'current'], 'TODO'
        assert isinstance(s.value, (list, tuple)), f'Trying to model as a function of {s}, but it has pinned value {s.value} instead of a range'
        assert len(s.value) == 2, f'Trying to model as a function of {s}, but it has value {s.value} instead of a range'

        if s.nominal != 0:
            #return center_expression_inputs(AnalogExpression, [s])(s, name)
            return AnalogExpression(s, name, centering_offset=s.nominal)
        else:
            return AnalogExpression(s, name)


def get_optional_expression_from_signals(s_list, name, all_signals):
    '''
    If s_list is empty, this is a constant.
    If s_list is not empty, it's a sum of expressions for each signal
    We need all_signals so we can tell what is an input when s is a string
    '''
    assert isinstance(name, str)
    individual = []
    for s in s_list:
        if isinstance(s, (SignalIn, SignalArray)):
            child_name = f'{name}_{s.friendly_name()}'
            individual.append(get_optional_expression_from_signal(s, child_name))
        elif isinstance(s, str):
            expr = get_expression_from_string(s, all_signals, f'{name}_<{s}>')
        else:
            assert False, f'Unrecognized optional expression type {type(s)} for "{s}"'
    if len(individual) == 0:
        individual.append(ConstExpression(f'{name}_const'))
    combiner = SumExpression(len(individual)+1, f'{name}_summer')
    total = HeirarchicalExpression(combiner, individual, name)
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

