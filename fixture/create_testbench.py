from collections import defaultdict
from numbers import Number

import pandas

import fixture
from itertools import product
import numpy as np
from fixture.signals import SignalIn, SignalOut, SignalArray, parse_bus, \
    parse_name


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
        assert test_vectors.shape[0] > 0, 'No test vectors passed to Testbench'
        self.test_vectors_all = test_vectors
        # TODO at this point, test_vectors.keys() has several things that we
        # want to ignore, but I don't know how to sort them out.
        # In particular, should we poke a whole SignalArray if its
        # individual bits are not in test_vectors?
        self.test_vectors = pandas.DataFrame({s: vs for s, vs in test_vectors.items()
                             if isinstance(s, SignalIn)})
        if self.test_vectors.shape == (0,0):
            # when we have no columns, self.test_vectors loses its length
            shape = (self.test_vectors_all.shape[0], 0)
            self.test_vectors = pandas.DataFrame(np.zeros(shape))
        self.do_optional_out = do_optional_out

    @staticmethod
    def scale_vector(vec, limits):
        def scale(val):
            return limits[0] + val * (limits[1] - limits[0])
        return [scale(val) for val in vec]

    def set_pinned_inputs(self):
        for s in self.test.signals:
            if not isinstance(s, (SignalIn, SignalArray)):
                continue
            if not s.auto_set:
                continue
            if not isinstance(s.value, Number):
                continue

            if isinstance(s, SignalIn):
                self.tester.poke(s.spice_pin, s.value)
            elif isinstance(s, SignalArray):
                bin_value = s.get_binary_value(s.value)
                for bit, val in zip(s, bin_value):
                    self.tester.poke(bit.spice_pin, val)

    def set_digital_mode(self, mode, delay_time=0):
        true_digital = self.test.signals.true_digital()
        for s, val in zip(true_digital, mode):
            self.tester.poke(s.spice_pin, val)
        self.tester.delay(delay_time)

    def apply_optional_inputs(self, i):
        for s in self.test_vectors.keys():
            if not s.auto_set:
                continue
            self.tester.poke(s.spice_pin, self.test_vectors[s][i])


    def read_optional_outputs(self):
        if not self.do_optional_out:
            return {}
        reads = {}
        for signal in self.test.signals.auto_measure():
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
                #assert s.template_name is not None, 'Not auto_set but not template? '+str(s)
                # can be neither auto-set nor template if it's an entry in a vectored input
                test_inputs[s] = self.test_vectors[s][i]
                if s.template_name is not None:
                    test_inputs[s.template_name] = self.test_vectors[s][i]
                if hasattr(s, 'spice_pin'):
                    test_inputs[s.spice_pin] = self.test_vectors[s][i]

        # Condense vectored inputs into a single entry in test_inputs
        for s in self.test.signals:
            if isinstance(s, SignalArray):
                # TODO I think maybe just ignore any and use all
                if any(entry in test_inputs for entry in s):
                    assert all(entry in test_inputs for entry in s)
                    # TODO could this be more than 1-D?
                    test_vec = np.array([test_inputs[entry] for entry in s])
                    test_inputs[s] = test_vec

        # look for string entries formatted like bus entries and create the
        # entire bus based on that
        # I wish we could just keep these organized as a bus since their
        # creation, but it's hard for input dimensions that are based on
        # vectored inputs
        bus_entries = defaultdict(dict)
        for t in test_inputs:
            if isinstance(t, str):
                bus, indices = parse_name(t)
                if len(indices) == 0:
                    continue
                bus_entries[bus][indices] = test_inputs[t]
        for bus, entries in bus_entries.items():
            dims = np.max(np.array(list(entries.keys())), 0) + 1
            bus_data = np.full(dims, np.nan)
            for loc, val in entries.items():
                bus_data[loc] = val
            test_inputs[bus] = bus_data

        reads_template = self.test.testbench(self.tester, test_inputs)
        reads_optional = self.read_optional_outputs()
        return (reads_template, reads_optional)
    
    def create_test_bench(self):
        self.result_processing_list = []
        self.set_pinned_inputs()

        true_digital = self.test.signals.true_digital()
        num_digital = len(true_digital)

        def convert_mode(mode_dict):
            assert set(true_digital) == set(mode_dict)
            return tuple(mode_dict[s] for s in true_digital)
        if len(self.template.digital_modes) == 0:
            self.true_digital_modes = list(product(range(2), repeat=num_digital))
        else:
            self.true_digital_modes = [convert_mode(md) for md in self.template.digital_modes]

        for digital_mode in self.true_digital_modes:
            self.set_digital_mode(digital_mode, self.template.extras.get('mode_settle_time', 0))
            for i in range(self.test_vectors.shape[0]):
                reads = self.run_test_point(i)
                self.result_processing_list.append((digital_mode, i, reads))

    def condense_results_analysis(self, results):
        # look at raw results to find input and output vectors
        # Output vectors: value changes between output modes
        # Input vectors: value is an ndarray
        # deal with results derived from vectored inputs/outputs
        if None in results:
            assert len(results) == 1
            results_out = results[None]
        else:
            # sort of want to transpose the order of the dicts
            output_vecs = list(results)
            names = list(results[output_vecs[0]])
            examples = {k: np.array(v) for k, v in results[output_vecs[0]].items()}
            results_out = {}
            for name in names:
                perfect_match = True
                for ov in output_vecs[1:]:
                    # issues with equality when the result is a list of ndarrays
                    result_ndarray = np.array(results[ov][name])
                    if not np.all(result_ndarray == examples[name]):
                        perfect_match = False
                        break

                # TODO had some issues with pole/zero extraction where things
                # happened to be a perfect match even though the user did not
                # expect it
                # TODO coming back to this later I don't remember exactly how it
                #  works ... seems like we were attempting to figure out which
                #  outputs from the analysis() method were vectored by seeing if
                #  they changed. But in more recent versions of the code we
                #  have that information stored on the test. But in the old
                #  version we have this "if False" so it's not doing that
                #  anyway ... I suspect there's a bug here where functions of
                #  the input (like input_abs) will be treated as vectored
                #  outputs even though test.input_mapping says they're not
                if False and perfect_match:
                    # doesn't change with respect to vectored output
                    results_out[name] = results[output_vecs[0]][name]
                else:
                    # does change, we need to vector name based on vectored output
                    for vec_i, ov in enumerate(output_vecs):
                        #name_vec = fixture.Regression.vector_parameter_name_output(name, vec_i, ov)
                        # TODO this would have to change if there were multiple
                        #  vectored outputs
                        name_vec = self.test.vectoring_dict[name][vec_i]
                        results_out[name_vec] = results[ov][name]

        # we've done output vectoring, now look for vectored inputs
        results_out_in = {}
        for name in results_out:
            entry = results_out[name][0]
            if isinstance(entry, np.ndarray) and len(entry.shape) > 0:
                # each entry is vectored
                parent_input_name = self.test.input_vector_mapping[name]
                parent_input = self.test.signals.from_template_name(parent_input_name)
                assert parent_input.shape == entry.shape, f'Vector length for {name} does not match {parent_input}'
                for vec_i, parent_input_entry in enumerate(parent_input):
                    data = [x[vec_i] for x in results_out[name]]
                    name_vec = fixture.Regression.vector_input_name(name, vec_i)
                    results_out_in[name_vec] = data
            else:
                results_out_in[name] = results_out[name]

        return results_out_in




    def call_analysis(self, reads_template):
        results_str = self.test.analysis(reads_template)
        if not isinstance(results_str, dict):
            assert False, 'Return from process_single_test should be a dict'


        promised_outputs_str = set(s.friendly_name() for s in self.test.analysis_outputs)
        for promised_analysis_output in promised_outputs_str:
            assert promised_analysis_output in results_str, f'analysis() method for {self.test} did not return any output for "{promised_analysis_output}", which is a required output'
        for returned_analysis_output in results_str.keys():
            assert returned_analysis_output in promised_outputs_str, f'Return from analysis() method for {self.test} included unexpeced item "{returned_analysis_output}"'

        results_sig = {}
        for key_str, val in results_str.items():
            key_sig = self.test.signals.from_template_name(key_str)
            results_sig[key_sig] = val

        return results_sig

    def get_results(self):
        ''' Return results in the following format:
        for mode: for [in, out]: {pin:[x1, x2, x3, ...], }
        '''
        # results_analysis holds any results from self.test.analysis()
        # results_analysis = {output_mode: {name: [value0, value1]}}
        # results_other holds the rest, which is optional reads and mode_id
        # results_other = {name: [value0, value1]}
        results_analysis = {}
        results_other = {}
        for loop_i, (m, result_i, (reads_template, reads_optional)) in enumerate(self.result_processing_list):
            # Note loop_i != result_i when there are multiple digital modes

            vectored_outputs = self.test.signals.vectored_out()
            if len(vectored_outputs) == 0:
                # no vectored output
                results_out_req = self.call_analysis(reads_template)
                results_out_req_vec = {None: results_out_req}
            else:
                # yes vectored output
                # TODO multiple vectored outputs is difficult. I think we would
                #  need to change self.test.vectoring_dict to keep track of
                #  where each component came from
                assert len(vectored_outputs) == 1, 'TODO multiple vectored outputs'
                vectored_output = vectored_outputs[0]
                results_out_req_vec = {}
                for vec_i, component in enumerate(vectored_output):
                    self.tester.set_vector_read_mode(vectored_output, component)
                    results_out_req = self.call_analysis(reads_template)
                    results_out_req_vec[component] = results_out_req
                self.tester.clear_vector_read_mode(vectored_output)


            results_out_opt = self.process_optional_outputs(reads_optional)

            # put results_analysis into lists
            if loop_i == 0:
                # first time; set up new dictionaries and lists
                for output_vec, result_dict in results_out_req_vec.items():
                    results_analysis[output_vec] = {}
                    for name, value in result_dict.items():
                        results_analysis[output_vec][name] = [value]
            else:
                for output_vec, result_dict in results_out_req_vec.items():
                    assert output_vec in results_analysis
                    for name, value in result_dict.items():
                        assert name in results_analysis[output_vec]
                        results_analysis[output_vec][name].append(value)

            # put results_other into list
            if loop_i == 0:
                # first time; set up new dictionaries and lists
                for name, value in results_out_opt.items():
                    results_other[name] = [value]
                results_other['mode_id'] = [m]
            else:
                for name, value in results_out_opt.items():
                    assert name in results_other
                    results_other[name].append(value)
                assert 'mode_id' in results_other
                results_other['mode_id'].append(m)

        results_analysis_vec = self.condense_results_analysis(results_analysis)
        num_modes = len(set(results_other['mode_id']))
        # careful - when v is a Series, each value is associated with an index,
        # and they will double up when you try to concatenate copies
        results_test_vectors = {k: pandas.concat([v]*num_modes, ignore_index=True)
                                for k, v in self.test_vectors_all.items()}


        results_comb = {**results_test_vectors,
                        **results_analysis_vec,
                        **results_other}
        results = pandas.DataFrame(results_comb)
        return results

    # TODO I don't think there's any reason why this should live in the
    #  testbench class. If we move it out, we wouldn't need to create the
    #  testbench to run the post processing step alone
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

