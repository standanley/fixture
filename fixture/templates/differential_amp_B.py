from fixture import TemplateMaster
from fixture.real_types import RealIn

class DifferentialAmpTemplate(TemplateMaster):
    required_ports = ['inp', 'inn', 'outp', 'outn']

    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'out_diff': {'gain':'in_diff', 'gain_from_cm':'in_cm', 'offset':'1'},
            'out_cm': {'gain_to_cm':'in_diff', 'cm_gain':'in_cm', 'cm_offset':'1'}
        }

        def input_domain(self):
            in_diff = RealIn(self.extras['limits_diff'], 'in_diff')
            in_cm = RealIn(self.extras['limits_cm'], 'in_cm')
            return [in_diff, in_cm]

        def testbench(self, tester, value):
            in_cm, in_diff = value['in_cm'], value['in_diff']
            tester.poke(self.ports.inp, in_cm + in_diff/2)
            tester.poke(self.ports.inn, in_cm - in_diff/2)
            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)
            #tester.expect(self.outp, 0, save_for_later=True)
            #tester.expect(self.outn, 0, save_for_later=True)

            readp = tester.get_value(self.ports.outp)
            readn = tester.get_value(self.ports.outn)
            return [readp, readn]


        def analysis(self, reads):
            outp = reads[0].value
            outn = reads[1].value
            return {'out_diff': outp - outn, 'out_cm': (outp + outn) / 2}

    tests = [Test1]

