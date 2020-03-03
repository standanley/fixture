from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput

class DifferentialAmpTemplate(TemplateMaster):
    required_ports = ['inp', 'inn', 'outp', 'outn']
    parameter_algebra = {
        'out_diff': {'gain':'in_diff', 'gain_from_cm':'in_cm', 'offset':'1'},
        'out_cm': {'gain_to_cm':'in_diff', 'cm_gain':'in_cm', 'cm_offset':'1'}
    }

    @classmethod
    def specify_test_inputs(self):
        in_diff = TestVectorInput(self.extras['limits_diff'], 'in_diff')
        in_cm = TestVectorInput(self.extras['limits_cm'], 'in_cm')
        return [in_diff, in_cm]

    @classmethod
    def run_single_test(self, tester, values):
        inp = values['in_cm'] + values['in_diff'] / 2
        inn = values['in_cm'] - values['in_diff'] / 2

        tester.poke(self.inp, inp)
        tester.poke(self.inn, inn)
        wait_time = float(self.extras['approx_settling_time'])*2
        tester.delay(wait_time)

        readp = tester.read(self.outp)
        readn = tester.read(self.outn)
        return [readp, readn]

    @classmethod
    def process_single_test(self, tester):
        outp = tester[0].value
        outn = tester[1].value

        results = {'out_diff': outp - outn, 
                   'out_cm':   (outp + outn) / 2}
        return results


    

