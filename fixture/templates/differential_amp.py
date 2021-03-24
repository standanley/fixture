from fixture import TemplateMaster

class DifferentialAmpTemplate(TemplateMaster):
    required_ports = ['inp', 'inn', 'outp', 'outn']


    class Test1(TemplateMaster.Test):
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
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
        }

        def input_domain(self):
            #inp = TestVectorInput(self.inp.limits, 'inp')
            #inn = TestVectorInput(self.inn.limits, 'inn')
            inp = self.signals.from_template_name('inp')
            inn = self.signals.from_template_name('inn')
            return [inp, inn]

        def testbench(self, tester, value):
            tester.poke(self.ports.inp, value['inp'])
            tester.poke(self.ports.inn, value['inn'])
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
            return {'outp': outp, 'outn': outn}
    
    tests = [Test1]
