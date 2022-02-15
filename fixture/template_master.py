import fault
from abc import ABC, abstractmethod
import fixture
from fixture import Tester
from fixture.signals import SignalManager, SignalArray, SignalOut
from fixture.plot_helper import PlotHelper

class TemplateMaster():

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

    def __init__(self, circuit, simulator, signal_manager, extras={}):
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

        # by the time the template is instantiated, a child should have added this
        assert hasattr(self, 'required_ports')
        self.check_required_ports()

        assert hasattr(self, 'tests')
        # replace test classes with instance
        self.tests = [T(self) for T in self.tests]

        for test in self.tests:
            test.signals = self.signals.copy()
            test_dimensions = test.input_domain()
            for s in test_dimensions:
                if isinstance(s, fixture.signals.SignalIn):
                    s.get_random = True
                    if s not in test.signals and s not in test.signals.flat():
                        test.signals.add(s)
                elif isinstance(s, fixture.signals.SignalArray):
                    for sig in s.flatten():
                        sig.get_random = True
                    if s not in test.signals:
                        test.signals.add(s)
                else:
                    assert False, 'input_domain must return SignalIn objects'

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
        def __init__(self, template):
            self.template = template
            #self.dut = template.dut
            self.ports = template.ports
            self.extras = template.extras
            self.debug_dict = {}
            # TODO this assert was removed at one point, but seems necessary?
            # I think it was to allow the algebra to be added later programatically?
            assert hasattr(self, 'parameter_algebra'), f'{self} should specify parameter_algebra!'

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

        def debug(self, tester, port, duration):
            '''
            This method will be overridden when the @debug decorator from
            template_creation_utils is added. Unfortunately this means this
            input signature has to match
            '''
            if isinstance(port, SignalOut):
                # probably represetnation, no way to do that now
                if hasattr(port, 'representation'):
                    port = port.representation['signal']
            port_name = str(self.template.signals.from_circuit_pin(port))
            if port_name not in self.debug_dict:
                r = tester.get_value(port, params={'style': 'block',
                                                   'duration': duration})
                self.debug_dict[port_name] = r

        def debug_plot(self):
            return
            import matplotlib.pyplot as plt
            plt.figure()
            leg = []
            bump = 0
            for p, r in self.debug_dict.items():
                leg.append(p)
                plt.plot(r.value[0], r.value[1] + bump, '-+')
                bump += 0.0 # useful for separating clock signals
            plt.grid()
            plt.legend(leg)
            plt.show()

        def __str__(self):
            s = repr(self)
            return s.split(' ')[0].split('.')[-1]


    def go(self, checkpoint, checkpoint_start=0):
        '''
        Actually do the entire analysis of the circuit
        '''

        checkpoint_controller = {str(test):
                                    {
                                        'choose_inputs': True,
                                        'run_sim': True,
                                        'run_regression': True,
                                        'run_post_regression': 'save'
                                    }
                                 for test in self.tests}

        #checkpoint_controller = {
        #    'StaticNonlinearityTest': {
        #        'run_sim': True,
        #        'run_regression': True,
        #        'run_post_regression': 'save'
        #    },
        #    'ChannelTest': {
        #        'run_sim': True,
        #        'run_regression': True,
        #        'run_post_regression': 'save'
        #    },
        #    'DelayTest': {
        #        'run_sim': True,
        #        'run_regression': True,
        #        'run_post_regression': 'save'
        #    },
        #    'SineTest': {
        #        'run_sim': True,
        #        'run_regression': True,
        #        'run_post_regression': 'save'
        #    },
        #    'ApertureTest': {
        #        'run_sim': True,
        #        'run_regression': True,
        #        'run_post_regression': 'save'
        #    },
        #    'KickbackTest': {
        #        'run_sim': True,
        #        'run_regression': True,
        #        'run_post_regression': 'save'
        #    }
        #}
        #checkpoint_controller = {
        #    'StaticNonlinearityTest': {
        #        'run_sim': False,
        #        'run_regression': False,
        #        'run_post_regression': False
        #    },
        #    'ChannelTest': {
        #        'run_sim': False,
        #        'run_regression': False,
        #        'run_post_regression': 'load'
        #    }
        #}

        params_by_mode_all = {}
        for test in self.tests:
            controller = checkpoint_controller[str(test)]

            if controller['choose_inputs']:
                test_vectors = fixture.Sampler.get_samples(
                    test.signals.random(),
                    getattr(test, 'num_samples', 10))
                checkpoint.save_input_vectors(test, test_vectors)

            if controller['run_sim']:
                tester = Tester(self.dut)
                # TODO what's a good way to specify do_optional_out
                do_optional_out = test == self.tests[0]

                test_vectors = checkpoint.load_input_vectors(test)
                tb = fixture.Testbench(self, tester, test, test_vectors,
                                       do_optional_out=do_optional_out)
                tb.create_test_bench()

                run_dir = checkpoint.suggest_run_dir(test)
                # TODO eventually we won't need no_run because we will just
                # skip this whole block instead (I think)
                self.simulator.run(tester, run_dir=run_dir, no_run=False)
                checkpoint.save_run_dir(test, run_dir)

                debug = True
                if debug:
                    test.debug_plot()

                results_each_mode = tb.get_results()
                checkpoint.save_extracted_data(test, results_each_mode)

            results_each_mode = checkpoint.load_extracted_data(test)
            params_by_mode = {}
            for mode in set(results_each_mode.mode_id):
                results = results_each_mode.loc[results_each_mode.mode_id==mode]
                if controller['run_regression']:
                    regression = fixture.Regression(self, test, results)

                    #PlotHelper.plot_regression(regression, test.parameter_algebra, regression.regression_dataframe)
                    #PlotHelper.plot_optional_effects(test, regression.regression_dataframe, regression.results)
                    rr = dict(regression.results)

                    checkpoint.save_regression_results(test, rr)
                else:
                    rr = {}

                if controller['run_post_regression'] == 'load':
                    with open(f'{test}_{mode}_post_regression.pickle', 'rb') as f:
                        import pickle
                        data_in = pickle.load(f)
                        test.post_regression(*data_in)
                elif controller['run_post_regression'] == 'save':
                    with open(f'{test}_{mode}_post_regression.pickle', 'wb') as f:
                        import pickle
                        pickle.dump((regression.results, regression.regression_dataframe), f)
                    # TODO this should really be handled in create_testbench
                    temp = test.post_regression(regression.results, regression.regression_dataframe)
                    rr.update(temp)
                elif controller['run_post_regression'] == False:
                    pass
                else:
                    assert False

                params_by_mode[mode] = rr

            # merge results from this test in results from all tests
            for mode in params_by_mode:
                if mode in params_by_mode_all:
                    params_by_mode_all[mode].update(params_by_mode[mode])
                else:
                    params_by_mode_all[mode] = params_by_mode[mode]

        return params_by_mode_all
