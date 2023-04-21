import fault
from abc import ABC, abstractmethod
import fixture
from fixture import Tester, Regression, sampler
from fixture.sampler import SampleStyle, get_sampler_for_signal
from fixture.signals import SignalManager, SignalArray, SignalOut, SignalIn, CenteredSignalIn
from fixture.plot_helper import PlotHelper

class TemplateMaster():
    debug = True

    class Ports:
        def __init__(self, signal_manager):
            self.sm = signal_manager

        def __len__(self):
            return len(self.sm.signals)

        def __getattr__(self, name):
            try:
                if name == '__len__':
                    assert False, 'unexpected, who is asking?'
                signals = self.sm.from_template_name(name)

                def get_spice(s):
                    return s.spice_pin if hasattr(s, 'spice_pin') else None
                ss = signals.map(get_spice) if isinstance(signals, SignalArray) else get_spice(signals)

                return ss
            except KeyError as err:
                raise AttributeError(err)

    def __init__(self, circuit, simulator, signal_manager, sample_groups, extras={}):
        '''
        circuit: The magma circuit
        port_mapping: a dictionary of {template_name: circuit_name} for required pins
        params: a dictionary of template-specific parameters
        '''

        self.signals = signal_manager
        self.ports = self.Ports(self.signals)
        self.dut = circuit
        self.extras = extras
        self.simulator = simulator
        self.sample_groups = sample_groups

        # by the time the template is instantiated, a child should have added this
        assert hasattr(self, 'required_ports')
        self.check_required_ports()

        assert hasattr(self, 'tests')
        # replace test classes with instance
        self.tests = [T(self) for T in self.tests]

    def required_port_info(self):
        # TODO: this should give more info than just the names of the ports
        # maybe expect the template creator to override this?
        return '\n'.join([str(port) for port in self.required_ports])

    def check_required_ports(self):
        '''
        Checks that the template instantiator actually mapped all the required ports.
        '''
        for port_name in self.required_ports:
            try:
                self.signals.from_template_name(port_name)
            except KeyError:
                assert False, 'Did not associate required port %s'%port_name

    '''
    Subclass Test will be used to organize methods related to tests
    '''
    class Test(ABC):
        # TODO perhaps put an init method that checks some parameter algebra
        # has been specified
        input_vector_mapping = {}

        def __init__(self, template):
            self.template = template
            self.ports = template.ports
            self.extras = template.extras
            self.debug_dict = {}

            # TODO this assert was removed at one point, but seems necessary?
            # I think it was to allow the algebra to be added later programatically?
            assert hasattr(self, 'parameter_algebra'), f'{self} should specify parameter_algebra!'


            self.signals = self.template.signals.copy()
            test_dimensions = self.input_domain()
            # TODO not sure this is the best way to do input signals
            self.input_signals = None # Todo get rid of this if unused
            self.sample_groups_test = []
            for x in test_dimensions:
                if isinstance(x, SampleStyle):
                    self.sample_groups_test.append(x)
                elif isinstance(x, (SignalIn, SignalArray)):
                    sg = get_sampler_for_signal(x)
                    self.sample_groups_test += sg
                else:
                    assert False, f'Return from Test.input_dimensions must be a list of SampleGroup or Signal, not list of {type(x)}'

            # We want test_dimensions to end up in self.signals, for parsing
            # parameter_algebra_vectored and possibly for other things too
            for x in test_dimensions:
                self.signals.add(x)
            self.signals.rebuild_ref_dicts()

            self._expand_parameter_algebra2()

            self.sample_groups_opt = template.sample_groups


        def _expand_parameter_algebra2(self):
            # Edit self.parameter_algebra to do many things:
            # Interpret algebra in the rhs strings
            # Replace strings with references to Signals
            # Expand input pins that are buses
            # Duplicate equations for vectored outputs
            # Replace input signals with centered versions where necessary
            # Put everything in "sum of products" form, i.e. dict of tuples
            #self.parameter_algebra_vectored = {k: v.copy() for k, v in self.parameter_algebra.items()}
            pa_vec = {}
            ones = ['1', Regression.one_literal]

            def read_algebra(algebra_string):
                # first, multiplication
                s = algebra_string.replace('**', '^')
                factors = s.split('*')
                ans = []
                for f in factors:
                    # now, powers
                    power_split = f.split('^')
                    if len(power_split) == 1:
                        ans.append(f)
                    elif len(power_split) == 2:
                        power = [power_split[0]]*int(power_split[1])
                        ans += power
                    else:
                        assert False, f'Double power in {algebra_string}?'
                return tuple(ans)


            # Copy and Interpret algebra in string
            for lhs, rhs in self.parameter_algebra.items():
                pa_vec[lhs] = {}
                for param, term in rhs.items():
                    if isinstance(term, str):
                        term_exp = read_algebra(term)
                        pa_vec[lhs][param] = term_exp
                    else:
                        pa_vec[lhs][param] = term

            def convert_string(string):
                # string to object
                if string in ones:
                    return Regression.one_literal
                try:
                    factor_obj = self.signals.from_template_name(string)
                    return factor_obj

                except KeyError:
                    # this term is not a template input
                    assert string not in self.template.required_ports, 'Unexpected case'
                    return string


            # convert to objects
            for lhs, rhs in pa_vec.items():
                for param, term in rhs.items():
                    # if the top level was a string, we still want it to be a
                    if isinstance(term, tuple):
                        term_objs = tuple(convert_string(s) for s in term)
                        pa_vec[lhs][param] = term_objs
                    else:
                        term_obj = convert_string(term)
                        pa_vec[lhs][param] = (term_obj,)


            # clean self.vector_mapping
            # TODO I think this can be done in Test.__init__, with better error handling, etc.
            vector_mapping_sig = {}
            if not hasattr(self, 'vector_mapping'):
                self.vector_mapping = {}
            for child_thing, parent_signals_str in self.vector_mapping.items():
                # child_thing can be a signal or a string used as a lhs
                parent_signals = [self.signals.from_template_name(name) for name in parent_signals_str]
                vector_mapping_sig[child_thing] = parent_signals

            # vector parameters (usually due to vectored inputs)
            # we do this in multiple passes in case there are products
            # of multiple vectored objects
            # first collect things to vector
            vectored_inputs = {}
            for lhs, rhs in pa_vec.items():
                for param, term in rhs.items():
                    for component in term:
                        vec_version = [component]

                        # when the component itself is vectored
                        if isinstance(component, SignalArray):
                            # vector this one
                            vec_version = list(component)

                        # when vector_mapping links the component to a vectored parent_input
                        for parent_input in vector_mapping_sig.get(component, []):
                            if isinstance(parent_input, SignalArray):
                                new_vec_version = []
                                for parent_i, parent_comp in enumerate(parent_input):
                                    rename_fun = (Regression.vector_parameter_name_input if isinstance(parent_comp, SignalIn)
                                                  else Regression.vector_parameter_name_output())
                                    parent_comp_vec = [rename_fun(comp, parent_i, parent_comp) for comp in vec_version]
                                    new_vec_version += parent_comp_vec
                                vec_version = new_vec_version

                        # if neither of those cases happened, then the length will still be 1
                        if len(vec_version) > 1:
                            vectored_inputs[component] = vec_version

            # now vector them one at a time
            for vectored_input, components in vectored_inputs.items():
                for lhs, rhs in pa_vec.items():
                    to_remove = []
                    to_add = {}
                    for param, term in rhs.items():
                        if vectored_input in term:
                            to_remove.append(param)
                            for vec_i, sub in enumerate(components):
                                term_subbed = tuple((sub if x == vectored_input else x)
                                                    for x in term)
                                param_name = Regression.vector_parameter_name_input(param, vec_i, sub)
                                # TODO is template_name the best choice here?
                                to_add[param_name] = term_subbed
                    for r in to_remove:
                        del pa_vec[lhs][r]
                    for a in to_add:
                        pa_vec[lhs][a] = to_add[a]

            # vector equations (usually due to vectored outputs, but can also happen when the LHS depends on
            # a vectored input)
            for lhs in list(pa_vec):
                if lhs in vector_mapping_sig:
                    rhs = pa_vec[lhs]
                    del pa_vec[lhs]
                    eqs = [(lhs, rhs)]
                    for vectored_output in vector_mapping_sig[lhs]:
                        if not isinstance(vectored_output, SignalArray):
                            continue
                        new_eqs = []
                        for vec_i, component in enumerate(vectored_output):
                            for lhs_j, rhs_j in eqs:
                                lhs_j_vec = Regression.vector_parameter_name_output(lhs_j, vec_i, component)
                                # we do need to rename params so they aren't confused
                                # in PlotHelper
                                rhs_j_renamed = {Regression.vector_parameter_name_output(param, vec_i, component): factors
                                               for param, factors in rhs_j.items()}
                                new_eqs.append((lhs_j_vec, rhs_j_renamed))
                        eqs = new_eqs

                    for new_lhs, new_rhs in eqs:
                        pa_vec[new_lhs] = new_rhs



            # optional outputs
            for s in self.signals.auto_measure():
                pa_vec[s.spice_name] = {f'{s.spice_name}_meas': (Regression.one_literal,)}

            self.parameter_algebra_vectored = pa_vec


        @abstractmethod
        def input_domain(self):
            '''
            Specify the input domain space for this test. Return a list of 
            Reals and Bits with names
            '''
            pass

        @abstractmethod
        def testbench(self, tester, values):
            '''
            Run a test for one operating point. Use the provided fault tester
            object and values dict containing a random value for each dimension
            in the specified input domain.
            '''
            pass

        @abstractmethod
        def analysis(self, reads):
            '''
            Given the GetValue objects from your testbench, convert things back
            to a nice domain for linear fitting to optional parameters.
            Return a dict with keys matching parameters and their measured values
            '''
            pass

        def post_regression(self, regression_models, regression_dataframe):
            '''
            After regression is run, this is called for additional post-processing.
            It it passed a dictionary with regression models {lhs: model}
            See model.model.data for the data used in the regression
            See model.predict() for the model's estimates for teh regression data
            See model.predict(new_data) to see the model's predictions on a new input
            Return a dict with additional parameters, same format as regression_models.results
            '''
            return {}

        def debug(self, tester, signal, duration):
            '''
            This method will be overridden when the @debug decorator from
            template_creation_utils is added. Unfortunately this means this
            input signature has to match
            '''
            if isinstance(signal, SignalArray):
                for signal in signal:
                    self.debug(tester, signal, duration)
                return
            if signal not in self.debug_dict:
                r = tester.get_value(signal, params={'style': 'block',
                                                   'duration': duration})
                self.debug_dict[signal] = r

        def debug_plot(self):
            from fixture.plot_helper import plt

            plt.figure()
            leg = []
            bump = 0
            for p, r in self.debug_dict.items():
                leg.append(p)
                plt.plot(r.value[0], r.value[1] + bump, '-+')
                bump += 0.0 # useful for separating clock signals
            plt.grid()
            plt.legend(leg)
            #plt.show()
            PlotHelper.save_current_plot(f'{self}_debug.png')
            #plt.savefig(f'{self}_debug.png', dpi=300)
            #plt.clf()


        def __str__(self):
            s = repr(self)
            return s.split(' ')[0].split('.')[-1]


    def go(self, checkpoint, checkpoint_controller):
        '''
        Actually do the entire analysis of the circuit
        '''


        params_by_mode_all = {}
        for test, controller in checkpoint_controller.items():

            if controller['choose_inputs']:
                test_vectors = fixture.Sampler.get_samples(test)
                checkpoint.save_input_vectors(test, test_vectors)

            # analysis requires a fault testbench even if we skip the actual
            # sim, so the checkpoint logic is not as straightforward here
            if controller['run_sim'] or controller['run_analysis'] or controller['run_post_process']:
                tester = Tester(self.dut)
                # TODO what's a good way to specify do_optional_out
                #do_optional_out = test == self.tests[0]
                do_optional_out = True

                test_vectors = checkpoint.load_input_vectors(test)
                tb = fixture.Testbench(self, tester, test, test_vectors,
                                       do_optional_out=do_optional_out)
                tb.create_test_bench()

                run_dir = checkpoint.suggest_run_dir(test)
                # even if we skip the sim, we still need fault to annotate all
                # the reads in the test bench, so we still need this call
                self.simulator.run(tester, run_dir=run_dir,
                                   no_run=(not controller['run_sim']))
                if controller['run_sim']:
                    checkpoint.save_run_dir(test, run_dir)

                if self.debug:
                    test.debug_plot()

                if controller['run_analysis']:
                    results_each_mode_unprocessed = tb.get_results()
                    checkpoint.save_extracted_data_unprocessed(test, results_each_mode_unprocessed)

            if controller['run_post_process']:
                results_each_mode_unprocessed = checkpoint.load_extracted_data_unprocessed(test)
                results_each_mode = tb.post_process(results_each_mode_unprocessed)
                checkpoint.save_extracted_data(test, results_each_mode)

            results_each_mode = checkpoint.load_extracted_data(test)
            results_each_mode[Regression.one_literal] = 1
            params_by_mode = {}
            for mode in set(results_each_mode.mode_id):
                results = results_each_mode.loc[results_each_mode.mode_id==mode]
                if controller['run_regression']:
                    regression = Regression(self, test, results)

                    print('TODO: plot optional effects - here or later?')
                    #ph.plot_optional_effects()

                    rr = regression.results_expr

                    checkpoint.save_regression_results(test, rr)
                    # TODO just a load test
                    rr = checkpoint.load_regression_results(test)
                else:
                    # TODO this is not working with modes
                    #  we should load it if/when we need it, I think
                    rr = checkpoint.load_regression_results(test)


                # TODO I'm getting rid of post_regression temporarily
                #temp = test.post_regression(regression.results_models, regression.regression_dataframe)
                #rr.update(temp)

                params_by_mode[mode] = rr


            # TODO this is temporary
            rr = checkpoint.load_regression_results(test)

            # PlotHelper.plot_regression(regression, test.parameter_algebra_vectored, regression.regression_dataframe)
            # PlotHelper.plot_optional_effects(test, regression.regression_dataframe, regression.results)
            # TODO this has not been tested with modes
            mode_prefix = '' if mode == '()' else f'mode_{mode}_'
            ph = PlotHelper(test,
                            test.parameter_algebra_vectored,
                            mode_prefix,
                            results_each_mode,
                            rr)
            #ph.plot_regression()
            ph.plot_results()

            # merge results from this test in results from all tests
            for mode in params_by_mode:
                if mode in params_by_mode_all:
                    params_by_mode_all[mode].update(params_by_mode[mode])
                else:
                    params_by_mode_all[mode] = params_by_mode[mode]

        return params_by_mode_all
