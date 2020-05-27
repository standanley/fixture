
def poke_binary_analog(tester, port, value):
    '''
    Assuming port is binary analog with port[0] as the
    low-order bit, poke "value" assuming value is scaled
    0 to 1.
    Note that if value is random, this just assigns a random value to each
    port independently, which is what we want to do for testing. Actually
    using "value" is helpful if the port happens to be binary-weighted
    and the value chosen with LHS or orthogonal sampling
    '''

    bits = len(port)
    int_value = int(value * 2**bits)
    for i in range(bits):
        tester.poke(port[i], int_value & 1)
        int_value >>= 1

def debug(test):
    class DebugTest(test):

        def debug(self, tester, port, duration):
            r = tester.get_value(port, params={'style':'block', 'duration': duration})
            self.debug_dict.append((port, r))

        def testbench(self, *args, **kwargs):
            self.debug_dict = []
            print('CREATED DEBUG DICT')
            retval = super().testbench(*args, **kwargs)
            return (self.debug_dict, retval)

        def analysis(self, reads):
            debug_dict, reads_orig = reads

            print('Doing analysis now')

            import matplotlib.pyplot as plt
            leg = []
            for p, r in debug_dict:
                leg.append(self.template.get_name_template(p))
                plt.plot(r.value[0], r.value[1], '-+')
            plt.grid()
            plt.legend(leg)
            plt.show()

            return super().analysis(reads_orig)

    return DebugTest