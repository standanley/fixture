import fault
import re
from itertools import product

# TODO timing

def scale_within_limits(limits, val):
    return limits[0] + val * (limits[1] - limits[0])

def add_vectors(tester, vectors):
    def poke(port_name, value):
        #p = re.compile('[^<>]+|<([0-9])+>')
        if port_name[-1] == '>':
            bus = re.match('[^<>]+', port_name).group()
            port = getattr(dut, bus)
            #print('port with full bus', port)
            for m in re.finditer('<([0-9]+)>', port_name):
                index = int(m.group(1))
                port = port[index]
        else:
            port = getattr(dut, port_name)
        #print('poking', port, value)
        tester.poke(port, value)


    # first .circuit gives you a CircuitWrapper
    dut = tester.circuit.circuit
    io = dut.IO

    assert len(dut.inputs_unspecified) == 0, f"Must specify acceptable inputs for {dut.inputs_unspecified}"

    for i in dut.inputs_pinned:
        port_name, pin = i
        #print('pinning', port_name, pin)
        poke(port_name, pin)

    num_digital = len(dut.inputs_digital)
    num_ranged = len(dut.inputs_ranged)
    input_vectors = []
    modes = product(range(2), repeat=num_digital)
    for mode, vectors_mode in zip(modes, vectors):
        # poke digital ports (set mode)
        for val, input_ in zip(mode, dut.inputs_digital):
                poke(input_, val)
        #num = sum(2**i*c for i,c in enumerate(mode))
        #print('setting ctrl to', num)
        #poke('ctrl', num)

        # loop through all the vectors for this mode
        for vec in vectors_mode:
            input_vec = list(mode)

            # poke analog ports
            vec_scaled = []
            for val, input_ in zip(vec[:num_ranged], dut.inputs_ranged):
                port_name, limits = input_
                val_ranged = scale_within_limits(limits, val)
                vec_scaled.append(val_ranged)
                poke(port_name, val_ranged) 
            input_vec += vec_scaled

            # poke binary analog ports
            for val, input_ in zip(vec[num_ranged:], dut.inputs_dai):
                poke(input_, val)
            input_vec += vec[num_ranged:]

            input_vectors.append(input_vec)

            #tester.eval()

            # read outputs
            outputs = dut.outputs_analog + dut.outputs_digital
            for out in outputs:
                port_name = out
                #print('expecting', port_name)
                # TODO support buses?
                port = getattr(dut, port_name)
                tester.expect(port, 0, save_for_later=True)


    def callback(tester):
        results_raw = tester.targets['spice'].saved_for_later
        i = 0
        results = []
        for input_vec in input_vectors:
            output_vec = []
            for out in outputs:
                output_vec.append(float(results_raw[i]))
                i += 1
            results.append((input_vec, output_vec))
        return results

    ranged_input_names = [x[0] for x in dut.inputs_ranged]
    inputs = dut.inputs_digital + ranged_input_names + dut.inputs_dai
    return ((inputs, outputs), callback)


