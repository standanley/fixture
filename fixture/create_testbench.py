import fixture
from itertools import product
from numpy import ndarray
from fixture.signals import SignalIn

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

    @staticmethod
    def scale_vector(vec, limits):
        def scale(val):
            return limits[0] + val * (limits[1] - limits[0])
        return [scale(val) for val in vec]

    def set_pinned_inputs(self):
        for s in self.test.signals:
            if isinstance(s, SignalIn) and s.auto_set:
                if not s.get_random:
                    assert isinstance(s.value, float) or isinstance(s.value, int), f'Unknown pin value for {s}'
                    self.tester.poke(s.spice_pin, s.value)

    def set_digital_mode(self, mode):
        true_digital = [s for s in self.test.signals if isinstance(s, SignalIn) and s.type_ == 'true_digital']
        for s, val in zip(true_digital, mode):
            self.tester.poke(s.spice_pin, val)

    def get_test_vectors(self):
        '''
        Set self.test_vectors to a dict of {signal: list}
        '''
        # TODO don't have a default; force test to set num_samples
        self.num_sample_points = getattr(self.test, 'num_samples', 10)
        all_signals = self.test.signals
        random_signals = [s for s in all_signals
            if isinstance(s, fixture.signals.SignalIn) and s.get_random]
        random_analog = [s for s in random_signals if s.type_ in ['analog', 'real']]
        random_ba = [s for s in random_signals if s.type_ in ['binary_analog', 'bit']]

        sample_points = fixture.Sampler.get_orthogonal_samples(
            len(random_analog),
            len(random_ba),
            self.num_sample_points
        )

        # Scale samples and put them into a dictionary
        # it's important that keys is all the analog then all the ba to match Sampler
        keys = random_analog + random_ba
        test_vectors = {}
        for i, s in enumerate(keys):
            sample_values_unscaled = [sp[i] for sp in sample_points]
            if s.type_ in ['analog', 'real']:
                assert type(s.value) == tuple and len(s.value) == 2, 'bad value '+str(s)
                sample_values = self.scale_vector(sample_values_unscaled, s.value)
            else:
                sample_values = sample_values_unscaled
            test_vectors[s] = sample_values

        self.test_vectors = test_vectors
        return

    def apply_optional_inputs(self, i):
        for s in self.test_vectors.keys():
            if not s.auto_set:
                continue
            self.tester.poke(s.spice_pin, self.test_vectors[s][i])


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

    def run_test_point(self, i):
        self.apply_optional_inputs(i)

        # TODO consider breaking the rest of this into another function
        test_inputs = {}

        # TODO instead of this, iterate through required ports, this way we get buses too
        for s in self.test_vectors.keys():
            if not s.auto_set:
                assert s.template_name is not None, 'Not auto_set but not template? '+str(s)
                test_inputs[s] = self.test_vectors[s][i]
                test_inputs[s.template_name] = self.test_vectors[s][i]
                if hasattr(s, 'spice_pin'):
                    test_inputs[s.spice_pin] = self.test_vectors[s][i]

        # TODO do we want to do this? we should be careful of bit ordering, but I think this is ok
        '''
        # turn lists of bits into magma BitVector types
        for name, val in test_inputs.items():
            if type(val) == list:
                test_inputs[name] = BitVector[len(val)](val)
        '''

        reads = self.test.testbench(self.tester, test_inputs)
        return reads
    
    def create_test_bench(self):
        self.get_test_vectors()
        self.result_processing_list = []
        self.set_pinned_inputs()

        true_digital = [s for s in self.test.signals if isinstance(s, SignalIn) and s.type_ == 'true_digital']
        num_digital = len(true_digital)
        self.true_digital_modes = list(product(range(2), repeat=num_digital))
        for digital_mode in self.true_digital_modes:
            self.set_digital_mode(digital_mode)
            #for v_optional, v_test in zip(self.optional_vectors, self.test_vectors):
            #    reads = self.run_test_vector(v_test, v_optional)
            #    self.result_processing_list.append((digital_mode, v_test, v_optional, reads))
            for i in range(self.num_sample_points):
                reads = self.run_test_point(i)
                self.result_processing_list.append((digital_mode, i, reads))

    def get_results(self):
        ''' Return results in the following format:
        for mode: for [in, out]: {pin:[x1, x2, x3, ...], }
        '''
        results_by_mode = {m: dict(self.test_vectors) for m in self.true_digital_modes}
        for m, i, reads in self.result_processing_list:
            results_out_req = self.test.analysis(reads)
            if not isinstance(results_out_req, dict):
                assert False, 'Return from process_single_test should be a dict'

            for k,v in results_out_req.items():
                if type(v) == ndarray:
                    results_out_req[k] = float(v)
            # TODO: optional outputs
            # results_out_opt = self.process_optional_outputs()

            for k,v in results_out_req.items():
                if k not in results_by_mode[m]:
                    assert i == 0, f'result {k} seen first at sample {i}'
                    results_by_mode[m][k] = []
                results_by_mode[m][k].append(v)

        # TODO there should maybe be a default implementation that does nothing?
        if hasattr(self.test, 'post_process'):
            for mode in results_by_mode:
                results_by_mode[mode] = self.test.post_process(results_by_mode[mode])

        self.results = [x for m,x in results_by_mode.items()]
        return self.results

