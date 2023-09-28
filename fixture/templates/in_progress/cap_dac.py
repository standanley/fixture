from fixture import TemplateMaster
import fixture
Regression = fixture.regression.Regression


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

        def post_regression(self, results, data):
            # TODO fix this
            def regression_name(s):
                return Regression.clean_string(Regression.regression_name(s))
            names = self.signals.from_template_name('in_').map(regression_name)

            vectors = data[names]
            therm = vectors.sum(1)
            measured_pos = data['outp']
            #predictions = model.predict()

            if False:
                import matplotlib.pyplot as plt
                plt.plot(therm, measured_pos, 'o')
                plt.xlabel('Thermometer code')
                plt.ylabel('Output voltage (measured)')
                #plt.plot(therm_prediction, predictions, 'x')
                plt.grid()
                plt.show()
            return {}

    tests = [Test1]

    

