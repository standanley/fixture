import fault
from abc import ABC, abstractmethod
import fixture
from fixture import Tester, Regression, sampler
from fixture.optional_fit import SympyExpression, get_expression_from_string
from fixture.sampler import SampleStyle, get_sampler_for_signal
from fixture.signals import SignalManager, SignalArray, SignalOut, SignalIn, \
    CenteredSignalIn, AnalysisResultSignal
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
                return signals

            except KeyError as err:
                raise AttributeError(err)

    def __init__(self, circuit, simulator, signal_manager, sample_groups, extras, digital_modes):
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
        self.digital_modes = digital_modes

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
        vector_mapping = {}
        bounds_dict = {}

        def __init__(self, template):
            self.template = template
            self.ports = template.ports
            self.extras = template.extras
            self.debug_dict = {}

            # TODO this assert was removed at one point, but seems necessary?
            # I think it was to allow the algebra to be added later programatically?
            assert hasattr(self, 'parameter_algebra'), f'{self} should specify parameter_algebra!'
            assert hasattr(self, 'analysis_outputs'), f'{self} must specify analysis_outputs'
            assert hasattr(self, 'parameters'), f'{self} must specify parameters'

            # inherited signals from template, just required pins I think
            self.signals = self.template.signals.copy()

            # signals whose values will be calculated by analysis() method
            analysis_outputs_str = self.analysis_outputs
            self.analysis_outputs = []
            for name in analysis_outputs_str:
                sig = AnalysisResultSignal(name)
                self.analysis_outputs.append(sig)
                self.signals.add(sig)

            test_dimensions = self.input_domain()
            self.input_signals = None # Todo get rid of this if unused
            self.sample_groups_test = []
            for x in test_dimensions:
                if isinstance(x, SampleStyle):
                    self.sample_groups_test.append(x)
                    for s in x.signals:
                        if s not in self.signals:
                            self.signals.add(s)
                elif isinstance(x, (SignalIn, SignalArray)):
                    sg = get_sampler_for_signal(x)
                    self.sample_groups_test += sg
                    if x not in self.signals:
                        self.signals.add(x)
                else:
                    assert False, f'Return from Test.input_dimensions must be a list of SampleGroup or Signal, not list of {type(x)}'

            # I feel like rebuild_ref_dicts isn't necessary here since we used
            # signals.add to add them, but I'm too scared to get rid of it
            self.signals.rebuild_ref_dicts()

            # for vector mapping, convert strings to signals
            if not hasattr(self, 'vector_mapping'):
                self.vector_mapping = {}
            vector_mapping_str = self.vector_mapping
            self.vector_mapping = {}
            for child_str, parents_str in vector_mapping_str.items():
                assert isinstance(child_str, str)
                assert all(isinstance(p, str) for p in parents_str)
                child = self.signals.from_template_name(child_str)
                parents = [self.signals.from_template_name(p) for p in parents_str]
                self.vector_mapping[child] = parents

            self.create_vectoring_dict()

            # vector analysis_outputs
            # TODO is checking for SignalArray the right way to identify
            #  vectored inputs, as far as test.vector_mapping is concerned?
            self.analysis_outputs_vectored = []
            for analysis_output in self.analysis_outputs:
                if analysis_output in self.vectoring_dict:
                    self.analysis_outputs_vectored += self.vectoring_dict[analysis_output]
                else:
                    self.analysis_outputs_vectored.append(analysis_output)

            self._expand_parameter_algebra2()

            self.sample_groups_opt = template.sample_groups

        def create_vectoring_dict(self):
            # sets self.vectoring_dict
            # vectoring_dict maps template signals to their vectored components
            # looks like {in: [in_diff, in_cm], pole: [pole_diff, pole_cm]}
            # remember that vectoring can be inherited by analysis outputs
            self.vectoring_dict = {}
            # when the component itself is vectored
            for s in self.signals:
                # only the template writer's vector_mapping is used to look
                # at this array, so only things with template names matter
                if s.template_name is not None and isinstance(s, SignalArray):
                    # Notice we use template name on the left (template writer
                    # is the one looking at this dict) but friendly name on the
                    # right (to create user-identifiable names)
                    self.vectoring_dict[s] = list(s)
            # when vector_mapping links the component to a vectored parent input
            for child, parents in self.vector_mapping.items():
                if not any(parent in self.vectoring_dict for parent in parents):
                    continue

                child_vector = [child]
                for parent in parents:
                    if parent not in self.vectoring_dict:
                        continue
                    child_vector_new = []
                    for child_v in child_vector:
                        new_signals = child_v.vector(self.vectoring_dict[parent])
                        for new_s in new_signals:
                            self.signals.add(new_s)
                        child_vector_new += new_signals
                    child_vector = child_vector_new

                self.vectoring_dict[child] = child_vector


        def _expand_parameter_algebra2(self):
            # Edit self.parameter_algebra to do many things:
            # Interpret algebra in the rhs strings
            # Replace strings with references to Signals
            # Expand input pins that are buses
            # Duplicate equations for vectored outputs
            # Replace input signals with centered versions where necessary
            # Put everything in "sum of products" form, i.e. dict of tuples
            ones = ['1', Regression.one_literal]

            self.parameter_algebra_vectored = {}
            for lhs, rhs in self.parameter_algebra.items():
                lhs_signal = self.signals.from_template_name(lhs)
                if lhs_signal in self.vectoring_dict:
                    for lhs_vec in self.vectoring_dict[lhs_signal]:
                        suffix = '_' + lhs_vec.friendly_name()
                        expr = get_expression_from_string(rhs, self.signals,
                                                          self.vectoring_dict,
                                                          lhs_vec.friendly_name(),
                                                          self.parameters,
                                                          param_suffix=suffix,
                                                          bounds_dict=self.bounds_dict)
                        self.parameter_algebra_vectored[lhs_vec] = expr
                else:
                    expr = get_expression_from_string(rhs, self.signals,
                                                      self.vectoring_dict,
                                                      lhs,
                                                      self.parameters,
                                                      bounds_dict=self.bounds_dict)
                    self.parameter_algebra_vectored[lhs_signal] = expr

            return




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
            PlotHelper.save_current_plot(f'{self}_debug')
            #plt.savefig(f'{self}_debug', dpi=300)
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
            modes = sorted(set(results_each_mode.mode_id))

            params_by_mode = {}
            if controller['run_regression']:
                for mode in modes:
                    mode_prefix = '' if mode == '()' else f'mode_{mode}_'
                    results = results_each_mode.loc[results_each_mode.mode_id==mode]
                    regression = Regression(self, test, results, mode_prefix)
                    rr = regression.results_expr
                    params_by_mode[mode] = rr

                checkpoint.save_regression_results(test, params_by_mode)
                # TODO just a load test
                params_by_mode = checkpoint.load_regression_results(test)

            else:
                # TODO this is not working with modes
                #  we should load it if/when we need it, I think
                params_by_mode = checkpoint.load_regression_results(test)


            # TODO I'm getting rid of post_regression temporarily
            #temp = test.post_regression(regression.results_models, regression.regression_dataframe)
            #rr.update(temp)



            # now for plots
            for mode in modes:
                mode_prefix = '' if mode == '()' else f'mode_{mode}_'
                results = results_each_mode.loc[
                    results_each_mode.mode_id == mode]
                rr = params_by_mode[mode]
                ph = PlotHelper(test,
                                test.parameter_algebra_final,
                                mode_prefix,
                                results,
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
