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
            #cls.sort_ports(cls)


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

#    def sort_ports(self):
#        # we want to sort ports into inputs/outputs/analog/digital/pinned/ranged, etc
#
#        for name, port in self.IO.items():
#            if port.isinput():
#                if isinstance(port, TupleKind):
#                    TODO
#                else:
#                    if isinstance(port, fault.RealKind):
#                        if hasattr(port, 'limits'):
#                            limits = port.limits
#                            try:
#                                pin = float(limits)
#                                inputs_pinned.append((port, pin))
#                            except TypeError:
#                                if len(limits) == 2:
#                                    inputs_ranged.append((port, tuple(limits)))
#                                else:
#                                    assert false
#
#                        print(name, 'is real')
#                    else:
#                        print(name, 'is not real')
#            elif port.isoutput():
#                pass
#            else:
#                # TODO deal with inouts?
#                raise NotImplemetedError()
