import fault

# TODO timing

def scale_within_limits(limits, val):
    return limits[0] + val * (limits[1] - limits[0])

def add_vectors(tester, vectors):

    # first .circuit gives you a CircuitWrapper
    dut = tester.circuit.circuit
    io = dut.IO

    assert len(dut.inputs_unspecified) == 0, f"Must specify acceptable inputs for {dut.inputs_unspecified}"

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


