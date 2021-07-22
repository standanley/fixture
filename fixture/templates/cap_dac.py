from fixture import TemplateMaster
from fixture.template_creation_utils import debug

class DACTemplate(TemplateMaster):
    required_ports = ['in_', 'outp', 'outn']

    #@debug
    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'outp': {'gainp': 'in_', 'offsetp': '1'},
            'outn': {'gainn': 'in_', 'offsetn': '1'}
        }
        required_info = {
            'approx_settling_time': 'Approximate time it takes for amp to settle within 99% (s)'
        }
        num_samples = 50

        def input_domain(self):
            return self.signals.from_template_name('in_')

        def testbench(self, tester, values):
            for p in self.ports.in_:
                self.debug(tester, p, 1)
            self.debug(tester, self.ports.outp, 1)
            self.debug(tester, self.ports.outn, 1)

            test = self.ports
            for bit in self.ports.in_:
                tester.poke(bit, values[bit])

            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)

            reads = {
                'outn': tester.get_value(self.ports.outn),
                'outp': tester.get_value(self.ports.outp)
            }
            return reads

        def analysis(self, reads):
            results = {k:float(v.value) for k,v in reads.items()}
            return results

        def post_regression(self, regression_models):
            model = list(regression_models.values())[0]
            data = model.model.data
            vectors = data.exog
            measured = data.endog
            predictions = model.predict()

            def get_therm(vector):
                # last entry is constant_one
                return sum(vector[:-1])
            therm_prediction = [get_therm(v) for v in vectors]

            #import matplotlib.pyplot as plt
            #plt.plot(therm_prediction, measured, 'o')
            #plt.plot(therm_prediction, predictions, 'x')
            #plt.grid()
            #plt.show()
            return {}

    tests = [Test1]

    

