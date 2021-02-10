from fixture import TemplateMaster

class OscillatorTemplate(TemplateMaster):
    required_ports = ['out']

    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'frequency': 'const'
        }
        required_info = {
            'approx_frequency': 'Ballpark guess for the output frequency (Hz)'
        }

        def input_domain(self):
            return []

        def testbench(self, tester, values):
            approx_period = 1 / float(self.extras['approx_frequency'])
            tester.delay(approx_period * 20)
            return tester.get_value(self.ports.out, params={'style':'frequency'})

        @classmethod
        def analysis(self, read):
            results = {'frequency': read.value}
            return results

    tests = [Test1]

