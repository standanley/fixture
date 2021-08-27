import fault
from abc import ABC, abstractmethod
import fixture
from fixture.signals import SignalManager, SignalArray
from fixture.plot_helper import PlotHelper

class TemplateMaster():

    class Ports:
        def __init__(self, signal_manager):
            self.sm = signal_manager

        def __len__(self):
            return len(self.sm.signals)

        def __getattr__(self, name):
            if name == '__len__':
                assert False, 'unexpected, who is asking?'
            signals = self.sm.template(name)

            def get_spice(s):
                return s.spice_pin if hasattr(s, 'spice_pin') else None
            ss = signals.map(get_spice) if isinstance(signals, SignalArray) else get_spice(signals)

            return ss

    def __init__(self, circuit, run_callback, signal_manager, extras={}):
        '''
        circuit: The magma circuit
        port_mapping: a dictionary of {template_name: circuit_name} for required pins
        params: a dictionary of template-specific parameters
        '''

        self.signals = signal_manager
        self.ports = self.Ports(self.signals)
        self.dut = circuit
        self.extras = extras
        self.run = run_callback

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
                self.signals.template(port_name)
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

        def post_regression(self, regression_models):
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
            pass
    

    def go(self):
        '''
        Actually do the entire analysis of the circuit
        '''
        params_by_mode_all = {}
        for test in self.tests:
            tester = fault.Tester(self.dut)
            tb = fixture.Testbench(self, tester, test)
            tb.create_test_bench()

            self.run(tester)

            results_each_mode = tb.get_results()

            params_by_mode = {}
            for mode, results in enumerate(results_each_mode):
                regression = fixture.Regression(self, test, results)

                PlotHelper.plot_regression(regression)
                PlotHelper.plot_optional_effects(test, results, regression.results)

                # TODO this should really be handled in create_testbench
                temp = test.post_regression(regression.results_models)

                rr = dict(regression.results)
                rr.update(temp)
                params_by_mode[mode] = rr

            # merge results from this test in results from all tests
            for mode in params_by_mode:
                if mode in params_by_mode_all:
                    params_by_mode_all[mode].update(params_by_mode[mode])
                else:
                    params_by_mode_all[mode] = params_by_mode[mode]

        return params_by_mode_all
