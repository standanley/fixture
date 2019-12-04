import fault
import re
from itertools import product
import collections
from hwtypes import BitVector

# TODO timing

def add_vectors():
    raise NotImplemented

class Testbench():
    def __init__(self, tester):
        self.tester = tester
        self.dut = tester.circuit.circuit
        self.io = self.dut.IO
        self.num_ba = len(self.dut.inputs_ba)
        self.num_ranged = len(self.dut.inputs_ranged)
        self.num_digital = len(self.dut.inputs_digital)


    def scale_vectors(self, vectors_unscaled):
        def scale(limits, val):
            return limits[0] + val * (limits[1] - limits[0])
        analog_limits = [port.limits for port in self.dut.inputs_ranged]
        ba_limits = [(0,1) for _ in self.dut.inputs_ba]
        lims = analog_limits + ba_limits
        vectors_scaled = []
        for vec in vectors_unscaled:
            scaled = [scale(lim, val) for lim,val in zip(lims, vec)]
            vectors_scaled.append(scaled)
        return vectors_scaled

    '''
    def add_vectors(tester, vectors):
        def poke(port_name, value):
            #p = re.compile('[^<>]+|<([0-9])+>')
            if port_name[-1] == '>':
                bus = re.match('[^<>]+', port_name).group()
                port = getattr(self.dut, bus)
                #print('port with full bus', port)
                for m in re.finditer('<([0-9]+)>', port_name):
                    index = int(m.group(1))
                    port = port[index]
            else:
                port = getattr(self.dut, port_name)
            #print('poking', port, value)
            tester.poke(port, value)


        # first .circuit gives you a CircuitWrapper
        dut = tester.circuit.circuit
        io = dut.IO

        # TODO fix the f string
        string = 'Must specify acceptable inputs for {dut.inputs_unspecified}'
        assert len(dut.inputs_unspecified) == 0, string

        for i in dut.inputs_pinned:
            port_name, pin = i
            #print('pinning', port_name, pin)
            poke(port_name, pin)

        num_digital = len(self.dut.inputs_digital)
        num_ranged = len(self.dut.inputs_ranged)
        input_vectors = []
        modes = product(range(2), repeat=num_digital)
        for mode, vectors_mode in zip(modes, vectors):
            # poke digital ports (set mode)
            for val, input_ in zip(mode, self.dut.inputs_digital):
                    poke(input_, val)
            #num = sum(2**i*c for i,c in enumerate(mode))
            #print('setting ctrl to', num)
            #poke('ctrl', num)

            # loop through all the vectors for this mode
            for vec in vectors_mode:
                input_vec = list(mode)

                # poke analog ports
                vec_scaled = []
                for val, input_ in zip(vec[:num_ranged], self.dut.inputs_ranged):
                    port_name, limits = input_
                    val_ranged = scale_within_limits(limits, val)
                    vec_scaled.append(val_ranged)
                    poke(port_name, val_ranged) 
                input_vec += vec_scaled

                # poke binary analog ports
                for val, input_ in zip(vec[num_ranged:], self.dut.inputs_dai):
                    poke(input_, val)
                input_vec += vec[num_ranged:]

                input_vectors.append(input_vec)

                #tester.eval()

                # read outputs
                outputs = self.dut.outputs_analog + self.dut.outputs_digital
                for out in outputs:
                    port_name = out
                    #print('expecting', port_name)
                    # TODO support buses?
                    port = getattr(dut, port_name)
                    tester.expect(port, 0, save_for_later=True)


        def callback(tester):
            results_raw = tester.targets['spice'].saved_for_later
            i = 0
            results = []
            for input_vec in input_vectors:
                output_vec = []
                for out in outputs:
                    output_vec.append(float(results_raw[i]))
                    i += 1
                results.append((input_vec, output_vec))
            return results

        ranged_input_names = [x[0] for x in dut.inputs_ranged]
        inputs = dut.inputs_digital + ranged_input_names + dut.inputs_dai
        return ((inputs, outputs), callback)
    '''

    '''
    def make_testbench(tester, N):
        dut = tester.circuit.circuit
        io = dut.IO
        # get random vectors
        if hasattr(dut, 'required_testbench_random_analog'):
            extra_analog = getattr(dut, 'required_testbench_random_analog')
        else:
            extra_analog = 0
    '''

    def set_pinned_inputs(self):
        for input_ in self.dut.inputs_pinned:
            val = input_.limits
            self.tester.poke(input_, val)

    def set_digital_mode(self, mode):
        for input_, val in zip(self.dut.inputs_digital, mode):
            self.tester.poke(input_, val)

    def set_test_vectors(self, vectors, prescaled = False):
        ''' Creates self.test_vectors_by_mode '''
        if (len(vectors) == 2**self.num_digital
                and len(vectors[0][0]) == self.num_ba + self.num_ranged):
            # TODO the line above used to have a +1, but I'm not sure why
            # maybe it was for the test_input
            modes = product(range(2), repeat=self.num_digital)
            self.test_vectors_by_mode = {}
            for mode, vectors_this_mode in zip(modes, vectors):
                if prescaled:
                    scaled = vectors_this_mode
                else:
                    scaled = self.scale_vectors(vectors_this_mode)
                self.test_vectors_by_mode[mode] = scaled
        else:
            # for now we only support a list of lists of input vectors
            raise NotImplementedError

    def is_optional(self, p):
        for optional in self.dut.optional_ports:
            # the whole point of this method is being able to use 
            # 'is' instead of '=='
            if p is optional:
                return True
        return False


    def apply_optional_inputs(self, test_vector):
        # poke analog ports
        zipped = zip(self.dut.inputs_ranged + self.dut.inputs_ba, test_vector)
        for input_, val in zipped:
            if self.is_optional(input_):
                self.tester.poke(input_, val) 

    def read_optional_outputs(self):
        # TODO use new read object
        # TODO think about test outputs being in the same list
        for port in self.dut.outputs_analog + self.dut.outputs_digital:
            self.tester.expect(port, 0, save_for_later = True)

    def process_optional_outputs(self):
        results = []
        for port in self.dut.outputs_analog + self.dut.outputs_digital:
            results.append(self.results_raw[self.result_counter])
            self.result_counter += 1
        return results

    def run_test_vector(self, test_vector):
        #num_required_a = len(self.dut.inputs_test_a)
        #num_optional_a = len(self.dut.inputs_ranged)
        #num_required_ba = len(self.dut.inputs_test_ba)
        #num_optional_ba = len(self.dut.inputs_ba)
        #v_required, v_optional = test_vector[:num_required], test_vector[num_required:]
        #abc
        #self.apply_optional_inputs(v_optional)
        #self.dut.run_single_test(self.tester, v_required)
        #self.read_optional_outputs()

        self.apply_optional_inputs(test_vector)

        # TODO consider breaking the rest of this into another function
        test_inputs = {}
        for input_, val in zip(self.dut.inputs_ranged, test_vector[:self.num_ranged]):
            if not self.is_optional(input_):
                name = type(input_).name
                test_inputs[name] = val

        for input_, val in zip(self.dut.inputs_ba, test_vector[self.num_ranged:]):
            if not self.is_optional(input_):
                # TODO I don't like manually taking everything after the '.',
                # but it looks like str() and repr() both give me the full name
                name = str(input_.name).split('.')[-1]
                bus_name = str(input_.name.array.name)
                arr = test_inputs.get(bus_name, []) + [val]
                test_inputs[bus_name] = arr
                test_inputs[name] = val

        for name, val in test_inputs.items():
            if type(val) == list:
                #test_inputs[name] = BitVector(val)
                test_inputs[name] = BitVector[len(val)](val)
        #print(test_inputs)
        self.dut.run_single_test(self.tester, test_inputs)

        return self.dut.process_single_test
    
    '''
    # Things for the user to override
    startup = lambda : None
    set_digital_mode = self.poke_digital_mode_inputs
    '''
        
    def create_test_bench(self):
        #self.startup()
        self.result_processing_list = []
        self.set_pinned_inputs()
        for digital_mode in self.test_vectors_by_mode:
            self.set_digital_mode(digital_mode)
            test_vectors = self.test_vectors_by_mode[digital_mode]
            for test_vector in test_vectors:
                callback = self.run_test_vector(test_vector)
                self.result_processing_list.append((digital_mode, test_vector, callback))

    def get_results(self):
        ''' Return results in the following format:
        for mode: for [in, out]: {pin:[x1, x2, x3, ...], }
        '''

        input_names, output_names = self.get_input_output_names()
        def append_vector(orig, data, pins):
            for x, p in zip(data, pins):
                orig[p] = orig.get(p, []) + [x]

        self.results_raw = self.tester.targets['spice'].saved_for_later
        self.results_raw = [float(x) for x in self.results_raw]
        results_by_mode = {m:({}, {}) for m in self.test_vectors_by_mode}
        self.result_counter = 0
        for m, v, fun in self.result_processing_list:
            result = fun(self)
            if not isinstance(result, collections.Sequence):
                result = [result]
                # TODO I think we should assert fail here rather than try to fix it
            result += self.process_optional_outputs()
            append_vector(results_by_mode[m][0], v, input_names)
            append_vector(results_by_mode[m][1], result, output_names)
        self.results = [x for m,x in results_by_mode.items()]
        print('Number of modes is', len(self.results))
        return self.results

    def get_input_output_names(self):
        inputs = self.dut.inputs_test + self.dut.inputs_ranged + self.dut.inputs_ba
        outputs = self.dut.outputs_test + self.dut.outputs_analog + self.dut.outputs_digital
        def clean(x):
            w = str(x)
            return w.split('.')[-1]
        input_names = [clean(x) for x in inputs]
        output_names = [clean(x) for x in outputs]
        print('returning', input_names, output_names)
        return input_names, output_names






