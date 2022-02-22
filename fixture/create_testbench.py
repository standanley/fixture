import pandas

import fixture
from itertools import product
from numpy import ndarray
from fixture.signals import SignalIn, SignalOut


def add_vectors():
    raise NotImplemented

class Testbench():
    def __init__(self, template, tester, test, test_vectors,
                 do_optional_out=False):
        '''
        tester: fault tester object
        test: TemplateMaster Test subclass object
        '''

        self.template = template
        self.tester = tester
        self.dut = tester.circuit.circuit
        self.test = test
        self.test_vectors = test_vectors
        self.do_optional_out = do_optional_out

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
        true_digital = self.test.signals.true_digital()
        for s, val in zip(true_digital, mode):
            self.tester.poke(s.spice_pin, val)

    def apply_optional_inputs(self, i):
        for s in self.test_vectors.keys():
            if not s.auto_set:
                continue
            self.tester.poke(s.spice_pin, self.test_vectors[s][i])


    def read_optional_outputs(self):
        if not self.do_optional_out:
            return {}
        reads = {}
        for signal in self.test.signals:
            if isinstance(signal, SignalOut) and signal.auto_measure:
                r = self.tester.get_value(signal)
                reads[signal.spice_name] = r
        return reads

    def process_optional_outputs(self, reads):
        # return dict with {str_name: float_val}
        results = {k: r.value for k, r in reads.items()}
        return results

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

        reads_template = self.test.testbench(self.tester, test_inputs)
        reads_optional = self.read_optional_outputs()
        return (reads_template, reads_optional)
    
    def create_test_bench(self):
        self.result_processing_list = []
        self.set_pinned_inputs()

        #true_digital = [s for s in self.test.signals if isinstance(s, SignalIn) and s.type_ == 'true_digital']
        true_digital = self.test.signals.true_digital()
        num_digital = len(true_digital)
        self.true_digital_modes = list(product(range(2), repeat=num_digital))
        for digital_mode in self.true_digital_modes:
            self.set_digital_mode(digital_mode)
            #for v_optional, v_test in zip(self.optional_vectors, self.test_vectors):
            #    reads = self.run_test_vector(v_test, v_optional)
            #    self.result_processing_list.append((digital_mode, v_test, v_optional, reads))
            for i in range(self.test.num_samples):
                reads = self.run_test_point(i)
                self.result_processing_list.append((digital_mode, i, reads))

    def get_results(self):
        ''' Return results in the following format:
        for mode: for [in, out]: {pin:[x1, x2, x3, ...], }
        '''
        results_by_mode = {m: dict(self.test_vectors) for m in self.true_digital_modes}
        for m, i, (reads_template, reads_optional) in self.result_processing_list:
            results_out_req = self.test.analysis(reads_template)
            if not isinstance(results_out_req, dict):
                assert False, 'Return from process_single_test should be a dict'

            for k,v in results_out_req.items():
                if type(v) == ndarray:
                    results_out_req[k] = float(v)

            results_out_opt = self.process_optional_outputs(reads_optional)
            # TODO this loop is copy/pasted from a few lines above
            for k,v in results_out_opt.items():
                if type(v) == ndarray:
                    results_out_opt[k] = float(v)

            for k,v in list(results_out_req.items()) + list(results_out_opt.items()):
                if k not in results_by_mode[m]:
                    assert i == 0, f'result {k} seen first at sample {i}'
                    results_by_mode[m][k] = []
                results_by_mode[m][k].append(v)

        results_chunks = []
        for mode, results_for_mode in results_by_mode.items():
            mode_id = sum(x*2**i for i, x in enumerate(mode[::-1]))
            N = len(next(iter(results_for_mode.values())))
            mode_col = [mode_id]*N
            data_mode = {'mode_id': mode_col, **results_for_mode}
            results_chunks.append(data_mode)
        results = pandas.concat(pandas.DataFrame(rc) for rc in results_chunks)
        return results

    def post_process(self, results):
        # run through post-processing and append new columns
        results_processed = results.copy()
        if hasattr(self.test, 'post_process'):
            for mode in set(results.mode_id):
                results_for_mode = results.loc[results.mode_id == mode]
                response = self.test.post_process(results_for_mode)

                if response is not None:
                    for new_column, values in response.items():
                        assert new_column not in results, f'Post-process column "{new_column}" already in results'
                        if new_column not in results_processed:
                            results_processed.insert(len(results_processed.columns),
                                                     new_column, float('nan'))
                        # TODO the commented version has a "chained indexing
                        # assignment" issue; I'm 90% sure that the uncommented
                        # version does the same thing but without that issue
                        #results_processed[new_column].loc[results.mode_id == mode] = values
                        results_processed.loc[results.mode_id == mode, new_column] = values

        return results_processed

