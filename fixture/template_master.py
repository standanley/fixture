from magma import *
import fault


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
        inputs_dai = []
        outputs_analog =[]
        outputs_digital = []

        for name, port in self.IO.items():
            if port.isinput():
                if isinstance(port, TupleKind):
                    TODO
                else:
                    if isinstance(port, fault.RealKind):
                        if hasattr(port, 'limits'):
                            limits = port.limits
                            if limits == None:
                                inputs_unspecified.append(name)
                            else:
                                try:
                                    pin = float(limits)
                                    inputs_pinned.append((name, pin))
                                except TypeError:
                                    if len(limits) == 2:
                                        #inputs_ranged.append((name, tuple(limits)))
                                        # magma overloads the name "tuple"
                                        inputs_ranged.append((name, limits))
                                    else:
                                        assert False
                        else:
                            inputs_unspecified.append(name)

                    elif isinstance(port, magma.BitKind):
                        inputs_digital.append(name)
                    else:
                        # maybe it's dai? 
                        TODO
            elif port.isoutput():
                if isinstance(port, TupleKind):
                    TODO
                else:
                    if isinstance(port, fault.RealKind):
                        outputs_analog.append(name)
                    elif isinstance(port, magma.BitKind):
                        outputs_digital.append(name)
                    else:
                        assert False, "Only analog and digital outputs are supported"

            else:
                # TODO deal with inouts?
                raise NotImplemetedError()
