from fixture import TemplateMaster

class DACTemplate(TemplateMaster):
    required_ports = ['in_', 'outp', 'outn']

    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'outp': {'gainp': 'in_', 'offsetp': '1'},
            'outn': {'gainn': 'in_', 'offsetn': '1'}
        }

        def input_domain(self):
            return self.ports.in_

        def testbench(self, tester, values):
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

    tests = [Test1]

    

