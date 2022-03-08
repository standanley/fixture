import os
import yaml
import fixture.cfg_cleaner as cfg_cleaner
from fixture import Representation
from fixture.signals import create_signal, parse_bus, parse_name, \
    SignalArray, SignalManager, SignalOut, SignalIn
import magma
import fault
import ast
import re
import numpy as np


def path_relative(path_to_config, path_from_config):
    ''' Interpret path names specified in config file
    We want path names relative to the current directory (or absolute).
    But we assume relative paths in the config mean relative to the config.
    '''
    if os.path.isabs(path_from_config):
        return path_from_config
    folder = os.path.dirname(path_to_config)
    res = os.path.join(folder, path_from_config)
    return res


def parse_test_cfg(test_config_filename_abs):
    with open(test_config_filename_abs) as f:
        test_config_dict = yaml.safe_load(f)
    if 'num_cycles' not in test_config_dict and test_config_dict['target'] != 'spice':
        test_config_dict['num_cycles'] = 10**9 # default 1 second, will quit early if $finish is reached
    return test_config_dict


def parse_extras(extras):
    for k, v in extras.items():
        if type(v) == str:
            try:
                v_parsed = ast.literal_eval(v)
                extras[k] = v_parsed
            except ValueError:
                # just leave it as a string
                pass
    return extras


def range_inclusive(s, e):
    d = 1 if e >= s else -1
    return list(range(s, e+d, d))




def parse_config(circuit_config_dict):

    #cfg_cleaner.edit_cfg(circuit_config_dict)


    # load test config data
    test_config_filename = circuit_config_dict['test_config_file']
    test_config_filename_abs = path_relative(circuit_config_dict['filename'], test_config_filename)
    test_config_dict = parse_test_cfg(test_config_filename_abs)

    # TODO this special case probably doesn't belong here
    if 'extras' in circuit_config_dict:
        if 'channel_info' in circuit_config_dict['extras']:
            channel_file_relative = circuit_config_dict['extras']['channel_info']['file_path']
            channel_file_abs = path_relative(circuit_config_dict['filename'], channel_file_relative)
            circuit_config_dict['extras']['channel_info']['file_path'] = channel_file_abs

    # go through pin definitions and create list of
    # [pin_info, circuit_name, template_name]
    # We don't create signals yet because we don't have template names and
    # create_signal has some checks involving that
    # Also put magma datatype into pin dict
    io = []
    signal_info_by_cname = {}
    pins = circuit_config_dict['pin']
    pin_params = ['datatype', 'direction', 'value', 'electricaltype', 'bus_type', 'first_one']
    digital_types = ['bit', 'binary_analog', 'true_digital']
    analog_types = ['real', 'analog']
    for name, p in pins.items():
        for pin_param in p.keys():
            assert pin_param in pin_params, f'Unknown pin descriptor {pin_param} for pin {name}'
        type_string = p['datatype']
        electrical_string = p.get('electricaltype', 'voltage')
        if type_string in digital_types:
            dt = magma.Bit
        elif type_string in analog_types:
            if electrical_string == 'current':
                dt = fault.ms_types.CurrentType
            else:
                dt = fault.ms_types.RealType
        else:
            assert False, f'datatype for {name} must be in {digital_types+analog_types}, not "{type_string}"'

        direction = p['direction']
        if direction == 'input':
            dt = magma.In(dt)
        elif direction == 'output':
            dt = magma.Out(dt)
        else:
            assert False, f'Direction for {name} must be "input" or "output", not "{direction}"'
        p['magma_datatype'] = dt

        bus_name, indices_parsed, info_parsed, bits = parse_bus(name)
        if len(indices_parsed) == 0:
            # dict, cname, tname
            signal_info_by_cname[name] = [p, name, None]
        else:
            # array of (dict, cname, tname)
            indices_limits = [max(i_range)+1 for i_range in indices_parsed]
            if bus_name in signal_info_by_cname:
                assert False, 'TODO'
            else:
                a = np.zeros(indices_limits, dtype=object)
                for bit_name, location in bits:
                    a[location] = [p, bit_name, None]
                signal_info_by_cname[bus_name] = a
    # still missing 2 things to create signals:
    # magma objects and template names

    # Now create the magma stuff
    for c_name, signal_info in signal_info_by_cname.items():
        if isinstance(signal_info, np.ndarray):
            dtypes = [info[0]['magma_datatype'] for info in signal_info.flatten()
                      if info is not None and 'magma_datatype' in info[0]]
            assert len(dtypes) > 0, f'Missing datatype for {c_name}'
            assert all(dtypes[0] == dt for dt in dtypes[1:]), f'Mismatched datatypes for {c_name}'
            np_shape = signal_info.shape
            magma_shape = np_shape[0] if len(np_shape) == 1 else np_shape[::-1]
            dt_array = magma.Array[magma_shape, dtypes[0]]
            io += [c_name, dt_array]
        else:
            p, c_name2, _ = signal_info
            assert c_name == c_name2, 'internal error in config_parse'
            io += [c_name, p['magma_datatype']]


    # now create proxy signals
    PROXY_SIGNAL_TAG = 'this_is_a_proxy_signal'
    if 'proxy_signals' in circuit_config_dict:
        test = circuit_config_dict['proxy_signals']
        for name, value_dict in test.items():
            assert name not in signal_info_by_cname, f'Duplicate name "{name}"'
            # list is [info_dict, circuit_name, template_name]
            # template name does not get assigned until later
            value_dict[PROXY_SIGNAL_TAG] = True
            # TODO mark all referenced signals as is_proxy_component
            signal_info_by_cname[name] = [value_dict, name, None]
            refs = Representation.get_referenced_signal_names(value_dict)
            for ref in refs:
                signal_info_by_cname[ref][0]['is_proxy_component'] = True

    ## TODO amp vector test stuff
    #signal_info_by_cname['inp'][0]['is_proxy_component'] = True
    #signal_info_by_cname['inn'][0]['is_proxy_component'] = True

    #checkpoint = Checkpoint()
    #UserCircuit = checkpoint.create_circuit(circuit_config_dict['name'], io)
    class UserCircuit(magma.Circuit):
        name = circuit_config_dict['name']
        IO = io

    # Now go through the template mapping and edit c_array to see template
    # names, also create t_array_entries_by_name with info by template name
    '''
    t: c
    t: c[0:7]
    t: c[7:0]
    t[0:7]: c
    t[7:0]: c
    t[7:0]: c[0:7]
    In which case(s) does t[0] map to c[0]?
    I think all cases except the last one
    '''

    t_array_entries_by_name = {}
    template_pins = circuit_config_dict['template_pins']
    for t, c in template_pins.items():
        t_bus_name, t_indices, t_info, _ = parse_bus(t)
        c_bus_name, c_indices, c_info, _ = parse_bus(c)
        # c_array is for the whole circuit bus, not just this entry/entries
        c_array = signal_info_by_cname[c_bus_name]

        # if bus is bus[0:2][0:3] and we assign to bus[1], we want c_indices
        # to look like [(1,), (0:3)], not just [(1,)]
        c_shape = getattr(c_array, 'shape', [])
        assert len(c_shape) >= len(c_indices), f'Too many indices in {c}'
        # extend c_indices so it goes all the way to the end of c
        c_indices += [(0, x-1) for x in c_shape[len(c_indices):]]

        # get back string form given indices; use the correct braces
        def get_t_name(t_indices_used):
            indices_text = [bs[0] + str(i) + bs[1] for i, bs in
                       zip(t_indices_used, t_info)]
            return t_bus_name + ''.join(indices_text)

        # match does 2 things:
        # edit c_array to insert template names
        # build up t_array_entries with template array entries
        if t_bus_name not in t_array_entries_by_name:
            t_array_entries_by_name[t_bus_name] = []
        t_array_entries = t_array_entries_by_name[t_bus_name]
        def match(t_indices_used, t_indices, c_a, c_indices):
            # if the template array is 1 entry, that entry encompasses the whole circuit array
            # if the template array is multiple entries, they must match with the circuit
            #    entries 1 to 1 until the template entries run out of dimensions
            #    if circuit runs out of dimensions first, that's an error
            # Dimensions that aren't a range are kinda skipped in this mapping
            #print('match called with', t_indices_used, t_indices, getattr(c_a, 'shape', []), c_indices)

            state_t = 0 if len(t_indices) == 0 else len(t_indices[0])
            state_c = 0 if len(c_indices) == 0 else len(c_indices[0])

            if state_t == 1:
                # descend on template single
                # keep t index in the "used" list but don't move on c
                match(t_indices_used + [t_indices[0][0]],
                      t_indices[1:],
                      c_a,
                      c_indices)
            elif state_c == 1:
                # descend on circuit single
                # not a range for c, so descend on c but don't move on t
                match(t_indices_used,
                      t_indices,
                      c_a[c_indices[0][0]],
                      c_indices[1:])

            elif state_t == 0 and state_c == 0:
                # we should be at the end of the array
                assert not isinstance(c_a, np.ndarray), 'internal error in config_parse?, should have extended c indices to match array'
                print('mapping', t_indices_used, c_a)
                t_name = get_t_name(t_indices_used)
                c_a[2] = t_name
                t_array_entries.append((t_indices_used, t_name))

            elif state_t == 0 and state_c == 2:
                # Add a dimension to t so it matches c
                # This doesn't descend yet, but will recurse to the 2,2 state
                match(t_indices_used,
                      t_indices + [c_indices[0]],
                      c_a,
                      c_indices)

            elif state_t == 2 and state_c == 0:
                assert False, 'Error matching {t}, {c}: {t} has too many dimensions'

            elif state_t == 2 and state_c == 2:
                # match indices
                # they both have ranges; they must match
                tr = range_inclusive(*t_indices[0])
                cr = range_inclusive(*c_indices[0])
                assert len(tr) == len(cr), f'Mismatched shapes for {t}, {c}'
                for ti, ci in zip(tr, cr):
                    match(t_indices_used + [ti],
                          t_indices[1:],
                          c_a[ci],
                          c_indices[1:])
            else:
                assert False, 'Internal error in config parse, unknown state'

        match([], t_indices, c_array, c_indices)


    # now actually create the signals
    def my_create_signal(c_info):
        pin_dict, c_name, t_name = c_info
        if PROXY_SIGNAL_TAG in pin_dict:
            return Representation.create_signal(c_name, t_name, pin_dict)
        else:
            # yes, this isn't the best place to edit the pin_dict, but it's okay
            value = ast.literal_eval(str(pin_dict.get('value', None)))
            pin_dict['value'] = value
            # TODO this needs to split the cname into parts
            bus_name, indices = parse_name(c_name)
            c_pin = getattr(UserCircuit, bus_name)
            for i in indices:
                assert isinstance(i, int), 'internal error in config_parse'
                c_pin = c_pin[i]
            return create_signal(pin_dict, c_name, c_pin, t_name)
    my_create_signal_vec = np.vectorize(my_create_signal)

    def extract_bus_info(c_info):
        info = {}
        acceptable = {'bus_type': ['any', 'thermometer', 'binary', 'one_hot'],
                      'first_one': ['low', 'high']}
        for pin_dict, c_name, _ in c_info.flatten():
            for k, v in pin_dict.items():
                # TODO these assertions are not what we actually want
                #assert k in acceptable, f'Unrecognized bus_info {k}, must be one of {list(acceptable.keys())}'
                #assert v in acceptable[k], f'Unrecognized option for bus_info {k}: {v}, must be one of {acceptable[k]}'
                # we have some bus info
                if k in info:
                    # must match
                    assert info[k] == v, f'Mismatched bus info in {c_name}'
                else:
                    info[k] = v
        return info

    signals = []
    for cn, c_info in signal_info_by_cname.items():
        if isinstance(c_info, np.ndarray):
            s_ndarray = my_create_signal_vec(c_info)
            bus_info = extract_bus_info(c_info)
            s_array = SignalArray(s_ndarray, bus_info, spice_name=cn)
            signals.append(s_array)
        else:
            s = my_create_signal(c_info)
            signals.append(s)


    # now put the signals into the template arrays
    # first create ndarrays of the correct size (this is kinda hard)
    signals_by_template_name = {}
    for t_bus_name, t_array_entries in t_array_entries_by_name.items():
        t_array_indices = list(zip(*t_array_entries))[0]
        t_array_limits = [max(indices)+1 for indices in zip(*t_array_indices)]
        signals_by_template_name[t_bus_name] = np.zeros(t_array_limits, dtype=object)

    # go through signals and place them in the right spots
    all_signals = ({*signals} | {*signals})
    #signals_flat = [s for s_or_a in signals for s in
    #                (s_or_a.flatten() if isinstance(s_or_a, SignalArray)
    #                 else [s_or_a])]
    for s in all_signals:
        if s.template_name is not None:
            t_bus_name, t_indices = parse_name(s.template_name)
            if len(t_indices) == 0:
                signals_by_template_name[s.template_name] = s
            else:
                a = signals_by_template_name[t_bus_name]
                a[t_indices] = s

    # and turn the ndarrays into SignalArrays
    for t_name, s_or_a in signals_by_template_name.items():
        if isinstance(s_or_a, np.ndarray):
            # TODO is there a way that this would need associated info?
            sa = SignalArray(s_or_a, {}, template_name=t_name)
            signals_by_template_name[t_name] = sa


    ## TODO TEMP FOR SAMPLER TEST
    #from fixture.signals import SignalOut
    #out_wrapper = SignalOut('real', None, None, 'out<0>')
    #old_out_0 = signals_by_template_name['out'][0]
    #out_wrapper.representation = {
    #    'reference': old_out_0,
    #    'signal': old_out_0
    #}
    #signals_by_template_name['out'].array[0] = out_wrapper
    ##signals[9].array[0] = out_wrapper
    #signals.append(out_wrapper)



    # now we just pack all our info into the signal manager
    sm = SignalManager(signals, signals_by_template_name)
    # TODO this is a bad spot to initialize proxy things I think
    # TODO I think we really do want to check entire arrays and their components
    for s in all_signals:
        if s.representation is not None:
            r = s.representation
            r.finish_init(sm)
            #r['reference'] = sm.from_circuit_name(r['reference'])

    template_class_name = circuit_config_dict['template']
    extras = parse_extras(circuit_config_dict['extras'])

    ## TODO ADDITIONAL SAMPLER TEST STUFF
    #clk_value = extras['clks'][old_out_0.spice_name]
    #del extras['clks'][old_out_0.spice_name]
    #extras['clks'][out_wrapper] = clk_value

    ## TODO VECTORED INPUT TEST STUFF
    #inp = SignalIn((0, 0.8), 'real', True, False, 'inp_spice', None, None, '')
    #inn = SignalIn((0, 0.8), 'real', True, False, 'inn_spice', None, None, '')
    #vector = np.array([inp, inn])
    #test = SignalArray(vector, {}, 'in_single', None, True)
    #sm.signals[2] = test
    #sm.signals_by_template_name['in_single'] = test
    #del sm.signals_by_circuit_name['my_in']


    return UserCircuit, template_class_name, sm, test_config_dict, extras



