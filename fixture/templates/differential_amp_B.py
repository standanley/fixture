from fixture import TemplateMaster
from fixture import TestVectorInput, TestVectorOutput

class DifferentialAmpTemplate(TemplateMaster):
    required_ports = ['inp', 'inn', 'outp', 'outn']
    parameter_algebra = {
        'out_diff': {'gain':'in_diff', 'cm_gain':'in_cm', 'offset':'1'},
        'out_cm': {'gain_to_cm':'in_diff', 'cm_gain_to_cm':'in_cm', 'offset_to_cm':'1'}
    }

    @classmethod
    def specify_test_inputs(self):
        # Calculate differential and cm limits given pin limits
        limits_p, limits_n = self.inp.limits, self.inn.limits
        limits_cm = (limits_p[0] + (limits_p[1] - limits_p[0]) * 0.25, 
            limits_p[0] + (limits_p[1] - limits_p[0]) * 0.75)
        limits_diff = ((limits_p[1] - limits_p[0]) * -0.5, 
                (limits_p[1] - limits_p[0]) * 0.5)

        print('limits cm', limits_cm, 'limits differential', limits_diff)
        in_diff = TestVectorInput(limits_diff, 'in_diff')
        in_cm = TestVectorInput(limits_cm, 'in_cm')
        return [in_diff, in_cm]

    @classmethod
    def run_single_test(self, tester, value):
        inp = value['in_cm'] + value['in_diff'] / 2
        inn = value['in_cm'] - value['in_diff'] / 2

        tester.poke(self.inp, inp)
        tester.poke(self.inn, inn)
        wait_time = float(self.extras['approx_settling_time'])*2
        tester.delay(wait_time)

        #tester.expect(self.outp, 0, save_for_later=True)
        #tester.expect(self.outn, 0, save_for_later=True)

        readp = tester.read(self.outp)
        readn = tester.read(self.outn)
        return [readp, readn]

    @classmethod
    def process_single_test(self, tester):
        outp = tester[0].value
        outn = tester[1].value

        results = {'out_diff': outp - outn, 
                'out_cm': (outp + outn) / 2}
        return results


    

