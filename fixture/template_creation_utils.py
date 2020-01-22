
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
