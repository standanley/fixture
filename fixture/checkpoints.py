import json

import jsonpickle
import magma
import fault
import jsonpickle

import fixture


class Checkpoint:

    def __init__(self):
        self.circuits_data = []

    def create_circuit(self, name_, io):
        # this is broken into its own method because of the way it will be
        # saved in jsonpickle as a class belonging to this method's locals
        # The root issue for this confusion is that magma circuits are classes,
        # and jsonpickle doesn't save classes
        # We also save the inputs so we can re-create this
        self.circuits_data.append((name_, io))

        class UserCircuit(magma.Circuit):
            name = name_
            IO = io

        return UserCircuit

    def save(self, thing, filename):
        # because jsonpickle will not store classes we have to save magma
        # circuits separately, and then restore them first and provide them
        # when we restore the rest
        combined = {}
        p1 = jsonpickle.Pickler(keys=True, warn=True)
        circuits_dict = p1.flatten(self.circuits_data)
        p2 = jsonpickle.Pickler(keys=True, warn=True)
        thing_dict = p2.flatten(thing)
        print(circuits_dict)
        combined = [circuits_dict, thing_dict]
        s = jsonpickle.json.encode(combined, indent=2)
        print(s)
        with open(filename, 'w') as f:
            f.write(s)
        print('done')

    def load(self, filename):
        with open(filename) as f:
            s = f.read()
            combined = jsonpickle.json.decode(s)
            circuits_dict, thing_dict = combined

            # first we re-create the magma io types
            # this is a bit hacky and depends on the structure of circuits_dict
            ios = [cd['py/tuple'][1] for cd in circuits_dict]
            type_strings = [x['py/type']
                            for io in ios for x in io if isinstance(x, dict)]

            def get_type(s):
                tokens = s.split('.')
                x = globals()
                x = x[tokens[0]]
                for token in tokens[1:-1]:
                    x = getattr(x, token)

                token, direction = tokens[-1][:-1].split('[')
                x = getattr(x, token)
                x = getattr(magma, direction)(x)
                return x
            io_types = [get_type(s) for s in type_strings]

            # now that we have the io types, re-create the user circuit
            p1 = jsonpickle.Unpickler(keys=True)
            circuits_data = p1.restore(circuits_dict, classes=io_types)
            circuits = [self.create_circuit(*cd) for cd in circuits_data]

            # now use the io types and user circuit to do the rest
            classes = circuits + io_types
            p2 = jsonpickle.Unpickler(keys=True)
            thing = p2.restore(thing_dict, classes=classes)

            return thing


class MyTest:
    pass




@jsonpickle.handlers.register(magma.DefineCircuitKind)#type(magma.Bit))
class MagmaCircuitHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj, data):
        print('in flatten', obj)
        pass

    def restore(self, obj):
        pass

def temp_getstate():
    print('in getstate')
    return {'test': 'getstate'}


if __name__ == '__main__':
    with open('pickletest2.json') as f:
        s = f.read()

    #class UserCircuit:
    #    name = 'bad_test'
    #    IO = ['bad_in', magma.In(magma.Bit)]

    fake_io = ['fake_in', magma.In(magma.Bit),
               'fake_out', magma.Out(magma.Bit),
               'fake_vdd', magma.In(fault.ms_types.RealType),
               'fake_vss', magma.In(fault.ms_types.RealType)
               ]
    x = fixture.config_parse.create_circuit('fake_name', fake_io)
    #x.UserCircuit = UserCircuit


    useful_type1 = magma.In(fault.ms_types.RealType)
    useful_type2 = type(x.fake_vdd)
    t = jsonpickle.decode(s, keys=True, classes=[x, useful_type1, useful_type2])

    exit()
    MyTest.important = 'Hi!'

    class MyCircuit(magma.Circuit):
        name = 'test'
        IO = ['input', magma.In(magma.Bit)]

    MyCircuit.__getstate__ = temp_getstate


    print(isinstance(MyCircuit, magma.DefineCircuitKind))
    x = [5, 'hi', (1,2,3), MyCircuit, 'x']
    s = jsonpickle.encode(x)
    print(s)
