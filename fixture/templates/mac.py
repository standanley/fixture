from fixture import TemplateMaster
from fixture import template_creation_utils

class MACTemplate(TemplateMaster):
    required_ports = ['d', 'W', 'outp', 'outn']

    @template_creation_utils.debug
    class Test1(TemplateMaster.Test):
        def __init__(self, *args, **kwargs):
            template = args[0]
            data = template.ports.d
            length = len(data)
            bit_width = len(data[0])

            linear_terms = {}
            for word in range(length):
                for data_bit in range(bit_width):
                    for weight_bit in range(bit_width):
                        suffix = self.get_prod_suffix(word, data_bit, weight_bit)
                        spice = template
                        linear_terms[f'c{suffix}'] = f'prod{suffix}'
            linear_terms['offset'] = '1'
            self.parameter_algebra = {
                'output_single_ended': linear_terms
            }

            num_terms = len(linear_terms)
            self.num_samples = num_terms * 2

            super().__init__(*args, **kwargs)

        #parameter_algebra = {
        #    'amp_output': {'dcgain': 'in_single', 'offset': '1'}
        #}

        def get_prod_suffix(self, word, data_bit, weight_bit):
            return f'_d{word}_{data_bit}_w{word}_{weight_bit}'

        def input_domain(self):
            d = [p for int4 in self.ports.d for p in int4]
            W = [p for int4
                 in self.ports.W for p in int4]
            return d + W

        def testbench(self, tester, values):
            self.debug(tester, self.ports.d[0][0], 1.0)
            self.debug(tester, self.ports.W[0][0], 1.0)
            self.debug(tester, self.ports.outn, 1.0)
            self.debug(tester, self.template.dut.z_debug, 1.0)

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

            tester.delay(1e-6)

            # TODO this makes way more sense in analysis, but I don't think I have access to values
            reads = {}
            for word in range(len(d_array)):
                for data_bit in range(len(d_array[0])):
                    for weight_bit in range(len(W_array[0])):
                        suffix = self.get_prod_suffix(word, data_bit, weight_bit)
                        data_val = values[d_array[word][data_bit]]
                        weight_val = values[W_array[word][weight_bit]]
                        prod = data_val * weight_val
                        reads[f'prod{suffix}'] = prod

            wait_time = float(self.extras['approx_settling_time'])*2
            tester.delay(wait_time)

            #reads = []
            #for out_word in self.ports.out:
            #    read_word = []
            #    for out_bit in out_word:
            #        read_word.append(tester.get_value(out_bit))
            #    reads.append(read_word)
            reads['outp'] = tester.get_value(self.ports.outp)
            reads['outn'] = tester.get_value(self.ports.outn)
            return reads

        def analysis(self, reads):
            ans = {}

            # copy all 1 bit partial products directly through
            for k, v in reads.items():
                if 'prod_' in k:
                    ans[k] = v

            # add actual measured output
            single_ended = reads['outp'].value - reads['outn'].value
            ans['output_single_ended'] = single_ended

            return ans

    tests = [Test1]

    

