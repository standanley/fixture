from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput

class DifferentialAmpTemplate(TemplateMaster):
    __name__ = 'DifferentialAmpTemplate'
    required_ports = ['inp', 'inn', 'outp', 'outn']
    parameter_algebra = ['I(outp - outn) ~ gain:I(inp-inn) + cm_gain:I((inp+inn)/2) + offset',
            'I((outp+outn)/2) ~ gain_to_cm:I(inp-inn) + cm_gain_to_cm:I((inp+inn)/2) + offset_to_cm']

    @classmethod
    def specify_test_inputs(self):
        inp = TestVectorInput(self.inp.limits, 'inp')
        inn = TestVectorInput(self.inn.limits, 'inn')
        return [inp, inn]

    @classmethod
    def specify_test_outputs(self):
        return [TestVectorOutput('outp'), TestVectorOutput('outn')]

    # TODO fix these last two methods
    @classmethod
    def run_single_test(self, tester, value):
        tester.poke(self.inp, value[0])
        tester.poke(self.inn, value[1])
        tester.expect(self.outp, 0, save_for_later=True)
        tester.expect(self.outn, 0, save_for_later=True)

    @classmethod
    def process_single_test(self, tester):
        results = []
        results.append(tester.results_raw[tester.result_counter])
        tester.result_counter += 1
        results.append(tester.results_raw[tester.result_counter])
        tester.result_counter += 1
        # for an amp, for now, no post-processing is required
        return results


    

