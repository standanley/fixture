from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput

class DifferentialAmpTemplate(TemplateMaster):
    required_ports = ['inp', 'inn', 'outp', 'outn']

    parameter_algebra = {
        'outp - outn': {
            'gain': 'inp-inn',
            'cm_gain': '(inp+inn)/2',
            'offset': '1'
        },
        '(outp+outn)/2': {
            'gain_to_cm': 'inp-inn',
            'cm_gain_to_cm': '(inp+inn)/2',
            'offset_to_cm': '1'
        }
    }

    @classmethod
    def specify_test_inputs(self):
        inp = TestVectorInput(self.inp.limits, 'inp')
        inn = TestVectorInput(self.inn.limits, 'inn')
        return [inp, inn]

    @classmethod
    def run_single_test(self, tester, value):
        tester.poke(self.inp, value['inp'])
        tester.poke(self.inn, value['inn'])
        wait_time = float(self.extras['approx_settling_time'])*2
        tester.delay(wait_time)
        tester.expect(self.outp, 0, save_for_later=True)
        tester.expect(self.outn, 0, save_for_later=True)

        readp = tester.read(self.outp)
        readn = tester.read(self.outn)
        return [readp, readn]


    @classmethod
    def process_single_test(self, tester):
        outp = tester[0].value
        outn = tester[1].value
        return {'outp': outp, 'outn': outn}

