import ast
from magma import *
import fault
from .real_types import BinaryAnalogType
from abc import ABC, abstractmethod
import fixture

class TemplateMaster():

    class Ports():
        def __init__(self, dut, mapping):
            self.dut = dut
            self.mapping = mapping

        def __getattr__(self, item):
            if item in self.mapping:
                return getattr(self.dut, self.mapping[item])
            else:
                super()
                #assert False, f'No required port named "{item}"'


    def __init__(self, circuit, port_mapping, run_callback, extras={}):
        '''
        circuit: The magma circuit
        port_mapping: a dictionary of {template_name: circuit_name} for required pins
        params: a dictionary of template-specific parameters
        '''

        self.dut = circuit
        self.mapping = port_mapping
        self.extras = extras
        self.run = run_callback
        self.ports = self.Ports(circuit, port_mapping)

        # by the time the template is instantiated, a child should have added this
        assert hasattr(self, 'required_ports')
        self.check_required_ports()

        self.reverse_mapping = {v:k for k,v in port_mapping.items()}

        assert hasattr(self, 'tests')
        # replace test classes with instance
        self.tests = [T(self) for T in self.tests]

        self.sort_ports()

    def go(self):
        '''
        Actually do the entire analysis of the circuit
        '''
        for test in self.tests:
            tester = fault.Tester(self.dut)
            tb = fixture.Testbench(self, tester, test)
            tb.create_test_bench()

            self.run(tester)

            results_each_mode = tb.get_results()
            for mode, results in enumerate(results_each_mode):
                regression = fixture.Regression(self, test, results)
                


    def required_port_info(self):
        # TODO: this should give more info than just the names of the ports
        # maybe expect the template creator to override this?
        return '\n'.join([str(port) for port in self.required_ports])

#     def is_required_port(self, p):
#         # TODO just use fixture_name?
#         # TODO I'm afraid it is not handling busses correctly, but maybe it doesn't matter?
#         # Single wires of an optional bus are counted as optional, which is all I need for now
#         required_mappings = [getattr(self, r).name for r in self.required_ports]
# 
#         return any(p.name == rn for rn in required_mappings)


    def check_required_ports(self):
        '''
        Checks that the template instantiator actually mapped all the required ports.
        '''
        for port_name in self.required_ports:
            assert port_name in self.mapping, 'Did not associate port %s'%port_name

    def get_name_template(self, p):
        try:
            return self.reverse_mapping[self.get_name_circuit(p)]
        except KeyError:
            assert False, f'Tried to get template name for non-template port {p}'

    def get_name_circuit(self, p):
        ''' gives back a string to identify something port-like
        The input could be a port type or port instance, etc.
        '''
        if type(p) == str:
            return p
        elif hasattr(type(p), 'name'):
            name = str(type(p).name)
            #print('FIRST CASE', name)
            return name
        elif isinstance(p, Type):
            name = str(p.name)
            #print('for ', p, 'trying', name)
            #print(self.is_required(p))
            #for required_port in self.required_ports:
            #    if name == str(getattr(self, required_port).name):
            #        name = required_port
            #        #print('matched! ', name)
            #        break
            name = name.split('.')[-1]
            #print('RETURING NAME', name)
            return name
        elif hasattr(p, 'name'):
            return p.name
        elif isinstance(p, fault.RealKind):
            print(p.name)
            raise NotImplementedError
        elif isinstance(p, Array):
            raise NotImplementedError
        elif issubclass(type(p), fault.RealKind):
            raise NotImplementedError
        else:
            print(p)
            print(type(p))
            print(p.name)
            raise NotImplementedError

    def sort_ports(self):

        circuit_port_names = [name for name, _ in self.dut.IO.items()]
        optional_port_names = [name for name in circuit_port_names if name not in self.mapping.values()]
        optional_ports = [getattr(self.dut, name) for name in optional_port_names]
        # optional_ports = []
        # for p in circuit_ports:
        #     # TODO is this the only time we really need get_name_circuit?
        #     name = self.get_name_circuit(p)
        #     # NOTE: can't use "in" on the next line because == doesn't work for ports
        #     if not any(p is port for port in self.mapping.values()):
        #         optional_ports.append(p)

        # we want to sort ports into inputs/outputs/analog/digital/pinned/ranged, etc
        inputs_pinned = []
        inputs_true_digital = []
        inputs_ba = []
        inputs_analog = []
        outputs_analog = []
        
        def sort_port(port):

            # recurse for arrays
            if isinstance(port, Array):
                for i in range(len(port)):
                    sort_port(port[i])
                return

            port_type = type(port)
            if port.isinout():
                raise NotImplementedError

            if not port.isinput():
                # NOTE: I'm not sure why I need the "not" above,
                # and magma.Flip does not work on port
                if isinstance(port_type, fault.RealKind):
                    assert hasattr(port, 'limits') and port.limits is not None, "Analog ports must have limits"
                    if type(port.limits) == str:
                        port.limits = ast.literal_eval(port.limits)

                    if type(port.limits) == float:
                        inputs_pinned.append(port)
                    elif hasattr(port.limits, '__len__') and len(port.limits) == 2:
                            inputs_analog.append(port)
                    else:
                        assert False, f'Cannot understand limits {port.limits} for {port}'

                elif issubclass(port_type, BinaryAnalogType):
                    inputs_ba.append(port)
                # TODO used to be BitKind on the next line
                elif isinstance(port, magma.Bit):
                    inputs_true_digital.append(port)
                else:
                    assert False, 'Cannot recognize port type {type(port)} for {port}'

            elif not port.isoutput():
                if isinstance(port_type, fault.RealKind):
                    outputs_analog.append(port)
                elif isinstance(port_type, magma.BitKind):
                    # No support yet for optional digital because of the nonlinearity
                    raise NotImplementedError
                    outputs_digital.append(port)
                else:
                    assert False, 'Cannot recognize port type {type(port)} for {port}'

            else:
                assert False, 'Cannot recognize input/output type {type(port)} for {port}'

        # Sort!
        for port in optional_ports:
            sort_port(port)

        ## Save results
        print('\nSaved results from port sorting:')
        print(inputs_pinned)
        print(inputs_true_digital)
        print(inputs_ba)
        print(inputs_analog)
        print(outputs_analog)
        #print(outputs_digital)


        self.inputs_pinned = inputs_pinned
        self.inputs_true_digital = inputs_true_digital
        self.inputs_ba = inputs_ba
        self.inputs_analog = inputs_analog
        self.outputs_analog = outputs_analog



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
    
