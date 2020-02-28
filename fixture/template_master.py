from magma import *
import fault
from .real_types import BinaryAnalogType
#from .real_types import BinaryAnalogKind, TestVectorOutput




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

            # set required port names 
            # TODO does this break things in magma?
            for port_name in cls.required_ports:
                getattr(cls, port_name).fixture_name = port_name

            # determine what random vectors might be needed to run a test
            assert hasattr(cls, 'specify_test_inputs'), 'Must specify required test inputs'
            cls.inputs_test = cls.specify_test_inputs()

            # sort ports into different lists depending on what kind of stimulus they require
            cls.sort_ports(cls)


        return cls

class TemplateMaster(Circuit, metaclass=TemplateKind):

    @classmethod
    def required_port_info(self):
        # TODO: this should give more info than just the names of the ports
        # maybe expect the template creator to override this?
        return '\n'.join([str(port) for port in self.required_ports])

    @classmethod
    def is_required_port(self, p):
        # TODO I'm afraid it is not handling busses correctly, but maybe it doesn't matter?
        # Single wires of an optional bus are counted as optional, which is all I need for now
        required_mappings = [getattr(self, r).name for r in self.required_ports]

        return any(p.name == rn for rn in required_mappings)

        # test_inputs_str = [self.get_name(x) for x in test_input_ports]
        # is_required_test = any(self.get_name(p) == rn for rn in test_inputs_str)

        # #print('checking required', p, is_required_port, is_required_test)
        # return is_required_port or is_required_test

    # gets called when someone subclasses a template, checks that all of
    # required_ports got mapped to in mapping
    def check_required_ports(self):
        for port_name in self.required_ports:
            assert hasattr(self, port_name), 'Did not associate port %s'%port_name

    @classmethod
    def get_name(self, p):
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
            for required_port in self.required_ports:
                if name == str(getattr(self, required_port).name):
                    name = required_port
                    #print('matched! ', name)
                    break
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


        circuit_ports = [getattr(self, name) for name, _ in self.IO.items()]

        def flip_test_input(ti):
            # magma flips ports when they are circuit inputs or outputs
            # it also deals in instances of the port type so we instantiate
            # we need to flip ones that magma hasn't already
            if ti not in circuit_ports:
                return ti.flip()()
            else:
                return ti
        test_input_ports = [flip_test_input(p) for p in self.inputs_test]

        self.optional_ports = [p for p in circuit_ports if not self.is_required_port(p)]

        # we want to sort ports into inputs/outputs/analog/digital/pinned/ranged, etc
        inputs_pinned = []
        inputs_true_digital = []
        inputs_optional = []
        inputs_required = []
        outputs_optional_analog = []

        temp_inputs_a_or_ba = []
        
        def sort_port(port):
            #if any(port == getattr(self, required) for required in self.required_ports):
            #if any(port.name == required for required in self.required_ports):
            #if any(port.name == rn for rn in required_mappings):
            #    # required ports don't go into these lists
            #    return
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
                    try:
                        pin = float(port.limits)
                        inputs_pinned.append(port)
                    except TypeError:
                        if len(port.limits) == 2:
                            #inputs_ranged.append((name, tuple(limits)))
                            # magma overloads the name "tuple"
                            # inputs_ranged.append(port)
                            temp_inputs_a_or_ba.append(port)
                        else:
                            # TODO put a better message here
                            assert False, 'Limits must be 1 or 2 values'

                #elif isinstance(port_type, BinaryAnalogKind):
                elif issubclass(port_type, BinaryAnalogType):
                    temp_inputs_a_or_ba.append(port)
                # TODO used to be BitKind on the next line
                elif isinstance(port, magma.Bit):
                    inputs_true_digital.append(port)
                else:
                    print('didint match any types')
                    print(port)
                    raise NotImplementedError
            elif not port.isoutput():
                if isinstance(port_type, fault.RealKind):
                    outputs_optional_analog.append(port)
                elif isinstance(port_type, magma.BitKind):
                    # No support yet for optional digital because of the nonlinearity
                    raise NotImplementedError
                    outputs_digital.append(port)
                else:
                    print(port, type(port), self.get_name(port))
                    assert False, "Only analog and digital outputs are supported"

            else:
                # TODO deal with unspecified input/output ?
                print('unspecified input/output')
                print(port)
                assert False

        # start sorting
        for port in self.optional_ports:
            sort_port(port)
        inputs_optional = temp_inputs_a_or_ba

        temp_inputs_a_or_ba = []
        for port in test_input_ports:
            sort_port(port)
        inputs_required = temp_inputs_a_or_ba

        #self.required_ba = inputs_ba[len(self.optional_ba):]
        #print('optional_a, optional_ba, required_ba')
        #print(self.optional_a)
        #print(self.optional_ba)
        #print(self.required_ba)

        ## Save results
        print('\nSaved results from port sorting:')
        print(inputs_pinned)
        print(inputs_true_digital)
        print(inputs_optional)
        print(inputs_required)
        print(outputs_optional_analog)
        #print(outputs_digital)


        self.inputs_pinned = inputs_pinned
        self.inputs_true_digital = inputs_true_digital
        self.inputs_optional = inputs_optional
        self.inputs_required = inputs_required
        self.outputs_optional_analog = outputs_optional_analog

        # self.inputs_pinned = inputs_pinned
        # self.inputs_ranged = inputs_ranged
        # self.inputs_unspecified = inputs_unspecified
        # self.inputs_digital = inputs_digital
        # self.inputs_ba = inputs_ba
        # self.outputs_analog = outputs_analog
        # self.outputs_digital = outputs_digital

