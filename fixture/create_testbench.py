import fixture
import fault
from itertools import product
from numpy import ndarray

import magma
from hwtypes import BitVector

# TODO timing

def add_vectors():
    raise NotImplemented

class Testbench():
    def __init__(self, template, tester, test):
        '''
        tester: fault tester object
        test: TemplateMaster Test subclass object
        '''

        self.template = template
        self.tester = tester
        self.dut = tester.circuit.circuit
        self.test = test

        #self.num_optional = len(self.dut.inputs_optional)
        #self.num_required = len(self.dut.inputs_required)
        #self.num_digital = len(self.dut.inputs_true_digital)

    '''
    def scale_vectors(self, vectors_unscaled):
        def scale(limits, val):
            return limits[0] + val * (limits[1] - limits[0])
        vectors_scaled = {}
        for port, values in vectors_unscaled.items():
            if hasattr(port, 'limits'):
                values_scaled = [scale(port.limits, val) for val in values]
            else:
                values_scaled = values
            vectors_scaled[port] = values_scaled
        return vectors_scaled
    '''

    def scale_vector(self, vec, limits):
        def scale(val):
            return limits[0] + val * (limits[1] - limits[0])
        return [scale(val) for val in vec]



    def set_pinned_inputs(self):
        for input_ in self.template.inputs_pinned:
            val = input_.limits
            self.tester.poke(input_, val)

    def set_digital_mode(self, mode):
        for input_, val in zip(self.template.inputs_true_digital, mode):
            self.tester.poke(input_, val)

    def get_test_vectors(self):
        '''
        Based on self.template and self.test, create test vectors
        for optional inputs and test inputs.
        Set self.optional_inputs, self.test_inputs as lists,
        then self.optional_vectors, self.test_vectors as lists.
        They are ordered such that zip(x_inputs, x_vectors[0]) would
        create tuples for the first test to run.
        '''
        # TODO how many samples?
        num_samples = 20
        # sort input_domain dimensions into analog and ba
        # test_analog = []
        # test_ba = []
        # for test_dim in self.test.input_domain():
        #     if hasattr(test_dim, 'limits'):
        #         test_analog.append(test_dim)
        #     else:
        #         test_ba.append(test_dim)
        test_analog = self.test.inputs_analog
        test_ba = self.test.inputs_ba

        # get random values with fixture.Sampler
        num_oa = len(self.template.inputs_analog)
        num_ta = len(test_analog)
        num_oba = len(self.template.inputs_ba)
        num_tba = len(test_ba)
        samples_T = fixture.Sampler.get_orthogonal_samples(
            num_oa + num_ta, num_oba + num_tba, num_samples)
        samples = list(zip(*samples_T))

        # reorganize samples into dictionary
        samples_oa_unscaled = samples[0:num_oa]
        samples_ta_unscaled = samples[num_oa:num_oa+num_ta]
        na = num_oa + num_ta
        samples_oba = samples[na:na+num_oba]
        samples_tba = samples[na+num_oba:]

        # scale
        samples_oa = []
        for p, vs in zip(self.template.inputs_analog, samples_oa_unscaled):
            samples_oa.append(self.scale_vector(vs, p.limits))
        samples_ta = []
        for p, vs in zip(test_analog, samples_ta_unscaled):
            samples_ta.append(self.scale_vector(vs, p.limits))

        self.optional_vectors = list(zip(*(samples_oa + samples_oba)))
        self.optional_inputs = self.template.inputs_analog + self.template.inputs_ba
        self.test_vectors = list(zip(*(samples_ta + samples_tba)))
        self.test_inputs = test_analog + test_ba

        if len(self.optional_vectors) == 0:
            self.optional_vectors = [()] * num_samples
        if len(self.test_vectors) == 0:
            self.test_vectors = [()] * num_samples
    """
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
    """

    def apply_optional_inputs(self, v_optional):
        zipped = zip(self.optional_inputs, v_optional)
        for input_, val in zipped:
            self.tester.poke(input_, val)

    ''' optional ouput not supported at the moment
    def read_optional_outputs(self):
        # TODO use new read object
        for port in self.dut.outputs_analog + self.dut.outputs_digital:
            assert False # not really supported right now :(
            r = self.tester.get_value(port)

    def process_optional_outputs(self):
        results = {}
        for port in self.dut.outputs_analog + self.dut.outputs_digital:
            if not self.dut.is_required(port):
                assert False # not really supported right now :(
                result = self.results_raw[self.result_counter]
                self.result_counter += 1
                results[self.dut.get_name_circuit(port)] = result
        return results
    '''

    def run_test_vector(self, vec_test, vec_optional):
        self.apply_optional_inputs(vec_optional)

        # TODO consider breaking the rest of this into another function
        test_inputs = {}

        # Add values with the ports as keys. Also note buses
        buses = set()
        for input_, val in zip(self.test_inputs, vec_test):
            test_inputs[input_] = val

            if isinstance(input_.name, magma.ref.ArrayRef):
                buses.add(input_.name.array)

        # Also add the bus as a whole
        for bus in buses:
            vals = []
            for port in bus.ts:
                vals.append(test_inputs[port])
            test_inputs[bus] = vals

        # in addition to using pins as keys, also use their string names
        for p, v in list(test_inputs.items()):
            # TODO is the next line ok?
            name = self.template.get_name_template(p)
            test_inputs[name] = v

        # turn lists of bits into magma BitVector types
        for name, val in test_inputs.items():
            if type(val) == list:
                test_inputs[name] = BitVector[len(val)](val)

        reads = self.test.testbench(self.tester, test_inputs)
        return reads
    
    '''
    # Things for the user to override
    startup = lambda : None
    set_digital_mode = self.poke_digital_mode_inputs
    '''
        
    def create_test_bench(self):
        #TODO set test vectors
        self.get_test_vectors()
        #self.startup()
        self.result_processing_list = []
        self.set_pinned_inputs()

        num_digital = len(self.template.inputs_true_digital)
        self.true_digital_modes = list(product(range(2), repeat=num_digital))
        for digital_mode in self.true_digital_modes:
            self.set_digital_mode(digital_mode)
            for v_optional, v_test in zip(self.optional_vectors, self.test_vectors):
                reads = self.run_test_vector(v_test, v_optional)
                self.result_processing_list.append((digital_mode, v_test, v_optional, reads))

    def get_results(self):
        ''' Return results in the following format:
        for mode: for [in, out]: {pin:[x1, x2, x3, ...], }
        '''
        results_by_mode = {m:{} for m in self.true_digital_modes}
        for m, req, opt, reads in self.result_processing_list:
            results_out_req = self.test.analysis(reads)
            if not isinstance(results_out_req, dict):
                results_out_req = [results_out_req]
                # TODO I think we should assert fail here rather than try to fix it
                assert False, 'Return from process_single_test should be a dict'

            for k,v in results_out_req.items():
                if type(v) == ndarray:
                    results_out_req[k] = float(v)
            # TODO: optional outputs
            # results_out_opt = self.process_optional_outputs()

            results_in_req = {k:v for k,v in zip(self.test_inputs, req)}
            results_in_opt = {k:v for k,v in zip(self.optional_inputs, opt)}

            results = {**results_in_req, **results_in_opt, **results_out_req}

            if len(results_by_mode[m]) == 0:
                results_by_mode[m] = {k:[] for k in results}

            for k,v in results.items():
                results_by_mode[m][k].append(v)


        self.results = [x for m,x in results_by_mode.items()]
        return self.results

