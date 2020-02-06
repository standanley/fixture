import fault
import re
from itertools import product
import collections

import magma
from hwtypes import BitVector
from magma import Array

# TODO timing

def add_vectors():
    raise NotImplemented

class Testbench():
    def __init__(self, tester):
        self.tester = tester
        self.dut = tester.circuit.circuit
        self.io = self.dut.IO
        # self.num_ba = len(self.dut.inputs_ba)
        # self.num_ranged = len(self.dut.inputs_ranged)
        self.num_optional = len(self.dut.inputs_optional)
        self.num_required = len(self.dut.inputs_required)
        self.num_digital = len(self.dut.inputs_true_digital)


    # def scale_vectors(self, vectors_unscaled):
    #     def scale(limits, val):
    #         return limits[0] + val * (limits[1] - limits[0])
    #     analog_limits = [port.limits for port in self.dut.inputs_ranged]
    #     ba_limits = [(0,1) for _ in self.dut.inputs_ba]
    #     lims = analog_limits + ba_limits

    #     vectors_scaled = {}
    #     for pin, vals in vectors_unscaled.items():
    #         limits = getattr(pin, 'limits', (0,1))
    #         vectors_scaled[pin] = [scale(limits, val) for val in vals]

    #     #print(vectors_unscaled)
    #     #print(vectors_scaled)
    #     #exit()
    #     

    #     # vectors_scaled = []
    #     # for vec in vectors_unscaled:
    #     #     scaled = [scale(lim, val) for lim,val in zip(lims, vec)]
    #     #     vectors_scaled.append(scaled)
    #     return vectors_scaled

    def scale_vectors(self, vectors_unscaled):
        def scale(limits, val):
            return limits[0] + val * (limits[1] - limits[0])
        vectors_scaled = {}
        for port, values in vectors_unscaled.items():
            if isinstance(type(port), fault.RealKind):
                values_scaled = [scale(port.limits, val) for val in values]
            else:
                values_scaled = values
            vectors_scaled[port] = values_scaled
        return vectors_scaled


    def set_pinned_inputs(self):
        for input_ in self.dut.inputs_pinned:
            val = input_.limits
            self.tester.poke(input_, val)

    def set_digital_mode(self, mode):
        for input_, val in zip(self.dut.inputs_true_digital, mode):
            self.tester.poke(input_, val)

    def set_test_vectors(self, vectors, prescaled = False):
        '''
        Creates self.test_vectors_by_mode
        vectors should be [for mode: {for port: [values]}]
        '''
        if (len(vectors) == 2**self.num_digital
                and len(vectors[0]) == self.num_optional + self.num_required):
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
            # for now we only support one way of specifying
            raise NotImplementedError

    def apply_optional_inputs(self, test_vector):
        zipped = zip(self.dut.inputs_optional, test_vector)
        for input_, val in zipped:
            self.tester.poke(input_, val)

    def read_optional_outputs(self):
        # TODO use new read object
        # TODO think about test outputs being in the same list
        for port in self.dut.outputs_analog + self.dut.outputs_digital:
            self.tester.expect(port, 0, save_for_later = True)

    def process_optional_outputs(self):
        results = {}
        for port in self.dut.outputs_analog + self.dut.outputs_digital:
            if not self.dut.is_required(port):
                result = self.results_raw[self.result_counter]
                self.result_counter += 1
                results[self.dut.get_name(port)] = result
        return results

    def run_test_vector(self, vec_required, vec_optional):
        self.apply_optional_inputs(vec_optional)

        # TODO consider breaking the rest of this into another function
        test_inputs = {}

        # Add values with the ports as keys. Also note buses
        buses = set()
        for input_, val in zip(self.dut.inputs_required, vec_required):
            #name = self.dut.get_name(input_)
            test_inputs[input_] = val
            if isinstance(input_.name, magma.ref.ArrayRef):
                buses.add(input_.name.array)

        # Also add each individual pin of the bus
        for bus in buses:
            vals = []
            for port in bus.ts:
                vals.append(test_inputs[port])
            test_inputs[bus] = vals

        # in addition to using pins as keys, also use their string names
        for p, v in list(test_inputs.items()):
            test_inputs[self.dut.get_name(p)] = v

        # turn lists of bits into magma BitVector types
        for name, val in test_inputs.items():
            if type(val) == list:
                test_inputs[name] = BitVector[len(val)](val)

        reads = self.dut.run_single_test(self.tester, test_inputs)
        return reads
    
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

            #TODO
            # should I put a dictionary of pin:value in the self.result_processing_list.append or should I put a list of [value] and also save a corresponding list of [pin]?
            #exit()
            #self.result_processing_input_pin_order = test_vectors.keys()
            #test_vectors_transpose = zip(*[test_vectors[p] for p in self.result_processing_input_pin_order])
            #print(list(test_vectors_transpose))
            def transpose(xs):
                return list(zip(*xs))
            test_vectors_required_T = transpose([test_vectors[p] for p in self.dut.inputs_required])
            test_vectors_optional_T = transpose([test_vectors[p] for p in self.dut.inputs_optional])

            if len(test_vectors_required_T) == 0:
                test_vectors_required_T = [() for _ in test_vectors_optional_T]
            if len(test_vectors_optional_T) == 0:
                test_vectors_optional_T = [() for _ in test_vectors_required_T]


            # print(list(test_vectors_optional_T))
            # print(list(test_vectors_required_T))

            for vec_required, vec_optional in zip(test_vectors_required_T, test_vectors_optional_T):
                reads = self.run_test_vector(vec_required, vec_optional)
                self.result_processing_list.append((digital_mode, vec_required, vec_optional, reads))

    def get_results(self):
        ''' Return results in the following format:
        for mode: for [in, out]: {pin:[x1, x2, x3, ...], }
        '''
        # input_names = [self.dut.get_name(p) for p in self.result_processing_input_pin_order]
        # def append_vector(orig, data, pins):
        #     for x, p in zip(data, pins):
        #         orig[p] = orig.get(p, []) + [x]
        # def append_vector(old, req, opt):
        #     for port, val in zip(ports, req + opt):
        #         old[port].append(val)

        self.results_raw = self.tester.targets['spice'].saved_for_later
        self.results_raw = [float(x) for x in self.results_raw]
        results_by_mode = {m:{} for m in self.test_vectors_by_mode}
        self.result_counter = 0
        for m, req, opt, reads in self.result_processing_list:
            results_out_req = self.dut.process_single_test(reads)
            if not isinstance(results_out_req, dict):
                results_out_req = [results_out_req]
                # TODO I think we should assert fail here rather than try to fix it
                assert False, 'Return from process_single_test should be a dict'

            # TODO: optional outputs
            # results_out_opt = self.process_optional_outputs()

            results_in_req = {k:v for k,v in zip(self.dut.inputs_required, req)}
            results_in_opt = {k:v for k,v in zip(self.dut.inputs_optional, opt)}

            results = {**results_in_req, **results_in_opt, **results_out_req}

            if len(results_by_mode[m]) == 0:
                results_by_mode[m] = {k:[] for k in results}

            for k,v in results.items():
                results_by_mode[m][k].append(v)


        self.results = [x for m,x in results_by_mode.items()]
        return self.results

