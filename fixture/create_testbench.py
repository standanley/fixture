import fault

# TODO timing

def scale_within_limits(limits, val):
    return limits[0] + val * (limits[1] - limits[0])

def add_vectors(tester, vectors):

    # first .circuit gives you a CircuitWrapper
    dut = tester.circuit.circuit
    io = dut.IO

    ## parse io to find inputs and outputs
    #inputs = [x for x in io.items() if x[1].isinput()]
    #inputs.sort()
    #outputs = [x for x in io.items() if x[1].isoutput()]
    #inputs_ranged, inputs_pinned = [], []
    #for i in inputs:
    #    assert hasattr(i[1], 'limits'), f'input {i[0]} is missing annotation of limits'
    #    #print(f'input {i} has range {i[1].limits}')
    #    if type(i[1].limits)==tuple:
    #       inputs_ranged.append((i, i[1].limits))
    #    else:
    #        inputs_pinned.append((i, i[1].limits))

    for i in dut.inputs_pinned:
        port_name, pin = i
        port = getattr(dut, port_name)
        tester.poke(port, pin)

    vectors_scaled = []
    for vec in vectors:
        vec_scaled = []
        for val, input_ in zip(vec, dut.inputs_ranged):
            port_name, limits = input_
            val_ranged = scale_within_limits(limits, val)
            vec_scaled.append(val_ranged)
            port = getattr(dut, port_name)
            tester.poke(port, val_ranged) 
        vectors_scaled.append(vec_scaled)

        #tester.eval()

        outputs = dut.outputs_analog + dut.outputs_digital
        for out in outputs:
            port_name = out
            port = getattr(dut, port_name)
            tester.expect(port, None, save_for_later=True)


    def callback(tester):
        results_raw = tester.targets['spice'].saved_for_later
        i = 0
        results = []
        for input_vec in vectors_scaled:
            output_vec = []
            for out in outputs:
                output_vec.append(results_raw[i])
                i += 1
            results.append((input_vec, output_vec))
        return results

    return ((dut.inputs_ranged, outputs), callback)


