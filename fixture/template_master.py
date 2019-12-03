from magma import *
import fault
from .real_types import BinaryAnalogKind
import re

class TemplateKind(circuit.DefineCircuitKind):


    def __new__(metacls, name, bases, dct):
        cls = super(TemplateKind, metacls).__new__(metacls, name, bases, dct)

        if name == 'TemplateMaster':
            # no checks are needed
            return cls

        is_template = (TemplateMaster in bases)
        if is_template:
            assert hasattr(cls, 'required_ports'), 'Template must give required ports'
            assert not hasattr(cls, 'mapping'), 'mapping is for the instance, not the template'

        else:
            assert hasattr(cls, 'mapping'), 'Subclass of template must provide port mapping'
            # call user's function to associate ports
            cls.mapping(cls)

            # check that all the required ports actually got associated
            cls.check_required_ports(cls)

            # determine what random vectors might be needed to run a test
            assert hasattr(cls, 'specify_test_inputs'), 'Must specify required test inputs'
            cls.inputs_test_a = []
            cls.inputs_test_ba = []
            for it in cls.specify_test_inputs():
                if it.binary_analog:
                    cls.inputs_test_ba.append(it)
                else:
                    cls.inputs_test_a.append(it)
            print('required stuff:')
            print(cls.inputs_test_a)
            print(cls.inputs_test_ba)


            # specify the names and number of outputs
            assert hasattr(cls, 'specify_test_outputs'), 'Must specify required test outputs'
            cls.outputs_test = cls.specify_test_outputs()

            # sort ports into different lists depending on what kind of stimulus they require
            cls.sort_ports(cls)


        return cls

class TemplateMaster(Circuit, metaclass=TemplateKind):

    @classmethod
    def required_port_info(self):
        # TODO: this should give more info than just the names of the ports
        # maybe expect the template creator to override this?
        return '\n'.join([str(port) for port in self.required_ports])


    # gets called when someone subclasses a template, checks that all of
    # required_ports got mapped to in mapping
    def check_required_ports(self):
        for port_name in self.required_ports:
            assert hasattr(self, port_name), 'Did not associate port %s'%port_name

    def sort_ports(self):
        # we want to sort ports into inputs/outputs/analog/digital/pinned/ranged, etc
        inputs_pinned = []
        inputs_ranged = []
        inputs_unspecified = []
        inputs_digital = []
        inputs_ba = []
        outputs_analog =[]
        outputs_digital = []

        required_mappings = [getattr(self, r).name for r in self.required_ports]

        def sort_port(port):
            #if any(port == getattr(self, required) for required in self.required_ports):
            #if any(port.name == required for required in self.required_ports):
            if any(port.name == rn for rn in required_mappings):
                # required ports don't go into these lists
                return
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
                    if hasattr(port, 'limits'):
                        limits = port.limits
                        if limits == None:
                            inputs_unspecified.append(port)
                        else:
                            try:
                                pin = float(limits)
                                inputs_pinned.append(port)
                            except TypeError:
                                if len(limits) == 2:
                                    #inputs_ranged.append((name, tuple(limits)))
                                    # magma overloads the name "tuple"
                                    inputs_ranged.append(port)
                                else:
                                    # TODO put a better message here
                                    assert False, 'Limits must be 1 or 2 values'
                    else:
                        # TODO I believe this case is not covered in the test
                        inputs_unspecified.append(port)

                elif isinstance(port_type, BinaryAnalogKind):
                    inputs_ba.append(port)
                elif isinstance(port_type, magma.BitKind):
                    inputs_digital.append(port)
                else:
                    print(port)
                    assert NotImplementedError
            elif not port.isoutput():
                if isinstance(port_type, fault.RealKind):
                    outputs_analog.append(port)
                elif isinstance(port_type, magma.BitKind):
                    outputs_digital.append(port)
                else:
                    assert False, "Only analog and digital outputs are supported"

            else:
                # TODO deal with unspecified input/output ?
                print(port)
                raise NotImplementedError

        for name, _ in self.IO.items():
            print('Sorting', name)
            port = getattr(self, name)
            sort_port(port)

        # Save results
        print('\nSaved results from port sorting:')
        print(inputs_pinned)
        print(inputs_ranged)
        print(inputs_unspecified)
        print(inputs_digital)
        print(inputs_ba)
        print(outputs_analog)
        print(outputs_digital)

        self.inputs_pinned = inputs_pinned
        self.inputs_ranged = inputs_ranged
        self.inputs_unspecified = inputs_unspecified
        self.inputs_digital = inputs_digital
        self.inputs_ba = inputs_ba
        self.outputs_analog = outputs_analog
        self.outputs_digital = outputs_digital

