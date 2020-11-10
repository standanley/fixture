from fixture import TemplateMaster

class MACTemplate(TemplateMaster):
    required_ports = ['d', 'W', 'out']

    class Test1(TemplateMaster.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'in_single', 'offset': '1'}
        }

        def input_domain(self):
            d = [p for int4 in self.ports.d for p in int4]
            W = [p for int4
                 in self.ports.W for p in int4]
            return d + W

        def testbench(self, tester, values):
            d_array = self.ports.d
            for d_word in d_array:
                assert len(d_word) == 4
                for d_bit in d_word:
                    tester.poke(d_bit, values[d_bit])

            W_array = self.ports.W
            for W_word in W_array:
                assert len(W_word) == 4
                for W_bit in W_word:
                    tester.poke(W_bit, values[W_bit])

            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)

            reads = []
            for out_word in self.ports.out:
                read_word = []
                for out_bit in out_word:
                    read_word.append(tester.get_value(out_bit))
                reads.append(read_word)
            return reads

        def analysis(self, reads):
            results = {'amp_output': reads.value}
            return results

    tests = [Test1]

    

