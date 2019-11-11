from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput

class DifferentialAmpTemplate(TemplateMaster):
    __name__ = 'DifferentialAmpTemplate'
    required_ports = ['inp', 'inn', 'outp', 'outn']
    parameter_algebra = ['I(outp - outn) ~ gain:I(inp-inn) + cm_gain:I((inp+inn)/2) + offset',
            'I((outp+outn)/2) ~ gain_to_cm:I(inp-inn) + cm_gain_to_cm:I((inp+inn)/2) + offset_to_cm']

    @classmethod
    def specify_test_inputs(self):
        # TODO figure out limits
        in_diff = TestVectorInput((0,1), 'in_diff')
        in_cm = TestVectorInput((0,1), 'in_cm')
        return [in_diff, in_cm]

    @classmethod
    def specify_test_outputs(self):
        return [TestVectorOutput('out_diff'), TestVectorOutput('out_cm')]


    # TODO fix these last two methods
    @classmethod
    def run_single_test(self, tester, value):
        inp = value[0] + value[1] / 2
        inn = value[0] - value[1] / 2
        tester.poke(self.inp, inp)
        tester.poke(self.inn, inn)
        tester.expect(self.outp, 0, save_for_later=True)
        tester.expect(self.outn, 0, save_for_later=True)

    @classmethod
    def process_single_test(self, tester):
        outp = tester.results_raw[tester.result_counter]
        tester.result_counter += 1
        outn = tester.results_raw[tester.result_counter]
        tester.result_counter += 1

        results = [outp - outn, (outp + outn) / 2]
        return results


    

