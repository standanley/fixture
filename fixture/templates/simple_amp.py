#from .. import templates
#from ..templates import *
from fixture import TemplateMaster

class SimpleAmpTemplate(TemplateMaster):
    __name__ = 'abc123'
    required_ports = ['in_single', 'out_single']

    @classmethod
    def run_single_test(self, tester):
        # For now we treat the input the same as the other analog inputs
        # so we don't need any pokes of our own
        #tester.expect(getattr(self, 'out_single'), 0, save_for_later=True)
        for output in self.outputs_analog:
            tester.expect(output, 0, save_for_later=True)

    @classmethod
    def process_single_test(self, tester):
        results = []
        for _ in self.outputs_analog:
            results.append(tester.results_raw[tester.result_counter])
            tester.result_counter += 1
        # for an amp, for now, no post-processing is required
        return results

