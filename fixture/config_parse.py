import itertools
import numbers
import os
import yaml
import fixture.cfg_cleaner as cfg_cleaner
from fixture import Representation, templates
from fixture.optional_fit import get_optional_expression_from_signals, \
    LinearExpression, HeirarchicalExpression
from fixture.sampler import SamplerConst, SamplerAnalog, get_sampler_for_signal
from fixture.signals import create_signal, parse_bus, parse_name, \
    SignalArray, SignalManager, SignalOut, SignalIn
import magma
import fault
import ast
import re
import numpy as np

from fixture.simulator import Simulator


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
    assert False
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


def create_magma_circuit(name_, circuit_definition_filename, physical_pin_signals):
    # TODO why do we need the file? Just to double-check, I think
    io = []
    for s in physical_pin_signals:
        if isinstance(s, SignalArray):
            # TODO is this the best way to do SA attributes?
            # s.type_ is inheriting the type from the individual bits,
            # which is a little confusing because the type of the sa itself
            # should be an array
            # TODO if we make s.type_ a magma array, we should do it in
            #  parse_physical_pins
            bit_type = s.spice_pin
            np_shape = s.shape
            magma_shape = np_shape[0] if len(np_shape) == 1 else np_shape[::-1]
            type_ = magma.Array[magma_shape, bit_type]
            io += [s.spice_name, type_]
        else:
            io += [s.spice_name, s.spice_pin]

    class UserCircuit(magma.Circuit):
        name = name_
        IO = io

    # now edit the signals to put the actual spice pins into spice_pin
    for s in physical_pin_signals:
        spice_pin = getattr(UserCircuit, s.spice_name)
        s.spice_pin = spice_pin
        if isinstance(s, SignalArray):
            assert len(s.array.shape) == 1, 'TODO multidimensional bus'
            for a, b in zip(s, spice_pin):
                a.spice_pin = b
    return UserCircuit


def parse_physical_pins(physical_pin_dict):
    # return a list of signals
    datatypes = {
        'bit': magma.Bit,
        'real': fault.ms_types.RealType,
        'current': fault.ms_types.CurrentType
    }
    directions = {
        'in': magma.In,
        'out': magma.Out
    }
    class BusInfo:
        def __init__(self, bus_name, loc, brackets, type_):
            self.bus_name = bus_name
            self.loc = loc
            self.brackets = brackets
            self.type_ = type_

    def make_signal(name, dt, direction, bus_info=None):
        if direction == 'in':
            return SignalIn(None, None, None, None, None, name, dt,
                                None, None, bus_info)
        elif direction == 'out':
            return SignalOut(None, name, dt, None, None, bus_info)
        else:
            assert False, f'Unknown direction "{direction}" for singal "{name}"'

    signals = []
    bus_bits = []
    bus_names = set()
    # A user might want to declare a bus in two pieces for some reason,
    # so we allow a sort of bit soup here and condense into buses later
    for name, info in physical_pin_dict.items():
        assert isinstance(info, dict), f'Issue with dict format for "{name}" physical pin'
        assert 'datatype' in info, f'Missing "datatype" key for "{name}" physical pin'
        assert info['datatype'] in datatypes, f'"datatype" for "{name}" must be one of {list(datatypes.keys())}'
        assert 'direction' in info, f'Missing "direction" key for "{name}" physical pin'
        assert info['direction'] in directions, f'"direction" for "{name}" must be one of {list(directions.keys())}'

        datatype = datatypes[info["datatype"]]
        direction = directions[info["direction"]]
        dt = direction(datatype)

        bus_name, indices_parsed, info_parsed, bits = parse_bus(name)
        if len(indices_parsed) == 0:
            signal = make_signal(name, dt, info['direction'])
            signals.append(signal)
        else:
            bus_names.add(bus_name)
            for bit_name, bit_pos in bits:
                type_ = info.get('type', None)
                bi = BusInfo(bus_name, bit_pos, info_parsed, type_)
                signal = make_signal(name, dt, info['direction'], bi)
                bus_bits.append(signal)

    # now go through each bus and collect its entries into an array
    for bus_name in bus_names:
        entries = [s for s in bus_bits if s.bus_info is not None and s.bus_info.bus_name == bus_name]
        assert len(entries) > 0
        brackets_rep = entries[0].bus_info.brackets
        assert all(e.bus_info.brackets == brackets_rep for e in entries), f'Mismatched number of dimensions, index direction, or bracket type for bus "{bus_name}"'
        type_rep = entries[0].bus_info.type_
        assert all(e.bus_info.type_ == type_rep for e in entries), f'Mismatched number of dimensions, index direction, or bracket type for bus "{bus_name}"'

        all_indices = np.array(list(e.bus_info.loc for e in entries))
        indices_limits_lower = np.min(all_indices, 0)
        for axis, lim in enumerate(indices_limits_lower):
            assert lim == 0, f'Lower limit for axis {axis} for bus "{bus_name}" must be 0, but it is {lim}'

        # like verilog, but not python, indices_limits_upper is INCLUSIVE
        indices_limits_upper = np.max(all_indices, 0)
        indices_limits_upper_exclusive = indices_limits_upper + 1
        array = np.empty(indices_limits_upper_exclusive, dtype=object)
        for e in entries:
            loc = e.bus_info.loc
            assert array[loc] is None, f'Duplicate definition of "{e.spice_name}"'
            array[loc] = e

        for loc in itertools.product(*list(range(b) for b in array.shape)):
            assert array[loc] is not None, f'Missing definition for "{bus_name}" at index {loc} for bus with shape {array.shape}'

        bi_total = BusInfo(bus_name, indices_limits_upper_exclusive, brackets_rep, type_rep)
        sa = SignalArray(array, bi_total, None, bus_name)
        signals.append(sa)

    sm = SignalManager(signals)
    return sm


def parse_proxy_signals(proxy_signal_dict, physical_signals):
    # create proxy signals return a list including the physical ones passed in
    #def signal_from_circuit_name(child_name, parent_name):
    #    ss = [s for s in signals if s.spice_name == child_name]
    #    assert len(ss) != 0, f'No existing signal "{child_name}" found for creating proxy signal {parent_name}'
    #    assert len(ss) < 2, f'Multiple signals with same name "{child_name}" when creating {parent_name}?'
    #    return ss[0]

    signals = physical_signals.copy()

    # now create proxy signals
    PROXY_SIGNAL_TAG = 'this_is_a_proxy_signal'
    for name, value_dict in proxy_signal_dict.items():
        assert name not in {s.spice_name for s in signals}, f'Duplicate name "{name}"'
        # list is [info_dict, circuit_name, template_name]
        # template name does not get assigned until later
        value_dict[PROXY_SIGNAL_TAG] = True
        # TODO mark all referenced signals as is_proxy_component
        #signal_info_by_cname[name] = [value_dict, name, None]
        Representation.convert_and_tag_referenced_signals(value_dict, signals)
        r = Representation.create_signal(name, None, value_dict)
        if r.representation is not None:
            r.representation.finish_init(signals)
        signals.add(r)
    return signals

def assign_template_pins(signals, template_matching):
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

    # TODO there's an issue if somebody assigns a slice of a bus to a template
    #  pin: what signal object do we return when the user asks for the template
    #  pin? I think we should check if the slice already exists, and if not,
    #  create a new SignalArray for that slice alone to give it the template name

    t_array_entries_by_name = {}
    template_pins = template_matching
    for t, c in template_pins.items():


        t_bus_name, t_indices, t_info, _ = parse_bus(t)
        c_bus_name, c_indices, c_info, _ = parse_bus(c)


        # TODO this is kind of a hack to write template names when the
        #  bus already exists. If the bus doesn't exist, we are in trouble
        #  also if the user is slicing the circuit bus right now, we are in trouble
        #  I think the best solution is to have rebuild_ref_dicts condense template bits
        if isinstance(signals.from_circuit_name(c), SignalArray):
            signals.from_circuit_name(c).template_name = t_bus_name

        # c_array is for the whole circuit bus, not just this entry/entries
        #c_array = signal_info_by_cname[c_bus_name]
        c_array = signals.from_circuit_name(c_bus_name).array

        # if bus is bus[0:2][0:3] and we assign to bus[1], we want c_indices
        # to look like [(1,), (0:3)], not just [(1,)]
        c_shape = getattr(c_array, 'shape', [])
        assert len(c_shape) >= len(c_indices), f'Too many indices in {c}'
        # extend c_indices so it goes all the way to the end of c
        c_indices += [(0, x-1) for x in c_shape[len(c_indices):]]

        # get back string form given indices; use the correct braces
        def get_t_name(t_indices_used):
            t_info_padded = t_info.copy()
            t_info_padded += ['[]a']*(len(t_indices_used) - len(t_info))

            indices_text = [bs[0] + str(i) + bs[1] for i, bs in
                            zip(t_indices_used, t_info_padded)]
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
                #print('mapping', t_indices_used, c_a)
                t_name = get_t_name(t_indices_used)
                c_a.template_name = t_name
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

    signals.rebuild_ref_dicts()
    return t_array_entries_by_name


def parse_stimulus_generation(signals, stim_dict):
    def samplers_from_entry(name, info):
        # never returns multiple, but sometimes returns zero
        try:
            s = signals.from_circuit_name(name)
            s.value = info

            if s.template_name is not None:
                # template writer has full control over this signal
                return []
            else:
                # must be optional input
                return get_sampler_for_signal(s)
        except KeyError:
            print('todo')

    sample_groups = []
    for name, info_str in stim_dict.items():
        info = ast.literal_eval(info_str)
        sample_groups += samplers_from_entry(name, info)
    return sample_groups


def get_simulator(circuit_config_dict):
    test_config_filename = circuit_config_dict['test_config_file']
    test_config_filename_abs = path_relative(circuit_config_dict['filename'], test_config_filename)
    with open(test_config_filename_abs) as f:
        test_config_dict = yaml.safe_load(f)
    if 'num_cycles' not in test_config_dict and test_config_dict['target'] != 'spice':
        test_config_dict['num_cycles'] = 10**9 # default 1 second, will quit early if $finish is reached

    circuit_filepath = circuit_config_dict['filepath']
    assert os.path.exists(circuit_filepath), f'Circuit filepath "{circuit_filepath}" not found'
    simulator = Simulator(test_config_dict, circuit_filepath)
    return simulator

def parse_config(circuit_config_dict):
    simulator = get_simulator(circuit_config_dict)

    physical_pin_signals = parse_physical_pins(circuit_config_dict['physical_pins'])
    UserCircuit = create_magma_circuit(
        circuit_config_dict['name'],
        circuit_config_dict['filepath'],
        physical_pin_signals
    )

    signals = parse_proxy_signals(circuit_config_dict['proxy_signals'], physical_pin_signals)

    template_mapping_unused = assign_template_pins(signals, circuit_config_dict['template_pins'])

    sample_groups = parse_stimulus_generation(signals, circuit_config_dict['stimulus_generation'])


    template_class_name = circuit_config_dict['template']
    TemplateClass = getattr(templates, template_class_name)

    extras = parse_extras(circuit_config_dict['extras'])

    t = TemplateClass(UserCircuit, simulator, signals, sample_groups, extras)

    return t


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
    pin_params = ['datatype', 'direction', 'value', 'electricaltype', 'bus_type', 'first_one', 'break_binary_for_bits']
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



    ## TODO amp vector test stuff
    #signal_info_by_cname['inp'][0]['is_proxy_component'] = True
    #signal_info_by_cname['inn'][0]['is_proxy_component'] = True

    #checkpoint = Checkpoint()
    #UserCircuit = checkpoint.create_circuit(circuit_config_dict['name'], io)
    class UserCircuit(magma.Circuit):
        name = circuit_config_dict['name']
        IO = io



    # now actually create the signals
    def my_create_signal(c_info):
        pin_dict, c_name, t_name = c_info
        # yes, this isn't the best place to edit the pin_dict, but it's okay
        if 'value' in pin_dict:
            value = ast.literal_eval(str(pin_dict.get('value', None)))
            pin_dict['value'] = value
        if PROXY_SIGNAL_TAG in pin_dict:
            return Representation.create_signal(c_name, t_name, pin_dict)
        else:
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
            # TODO nested buses
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
    all_signals = ({*signals} |
                   {s for a in signals if isinstance(a, SignalArray) for s in a })
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

    optional_input_info = circuit_config_dict.get('optional_input_info', {})


    return UserCircuit, template_class_name, sm, test_config_dict, optional_input_info, extras


def parse_optional_input_info(circuit_config_dict, tests):
    # TODO this can't be in the main parse_config because it needs to wait
    # until after the template has expanded parameter_algebra
    # TODO I think we should pass in the list of tests so we can look through
    # the parameter_algebra_vectored for each one

    optional_input_info = circuit_config_dict.get('optional_input_info', {})

    # we only make this list of params for an error message
    params = []

    # we create a copy and then delete entries as we use them to find unused
    info_copy = optional_input_info.copy()
    for test in tests:
        parameter_algebra_expr = {}
        for lhs, rhs in test.parameter_algebra_vectored.items():
            rhs_new = {}
            for param, multiplier in rhs.items():
                params.append(param)
                #params[param] = (test, lhs)
                # get optional expression for param
                signals = None
                if param in info_copy:
                    # todo
                    signals_str = info_copy[param]
                    del info_copy[param]
                    assert isinstance(signals_str,
                                      list), f'Optional input dependencies for {param} should be list, not {effects}'
                    signals = [test.signals.from_circuit_name(s_str)
                               for s_str in signals_str]
                else:
                    # default expr
                    signals_with_arrays = test.signals.optional_expr()
                    #signals = [s for x in signals_with_arrays for s in (
                    #    x if isinstance(x, SignalArray) else [x])]
                    signals = signals_with_arrays
                    print(f'Using default effect model for {param}, which includes {signals}')

                exp = get_optional_expression_from_signals(signals, param)
                rhs_new[exp] = multiplier
            parameter_algebra_expr[lhs] = rhs_new
        test.parameter_algebra_expr = parameter_algebra_expr


    if len(info_copy) > 0:
        raise KeyError(
            f'Specified optional effect(s) for {list(info_copy.keys())}, but recognized params are {params}')

    # now combine the expressions from each parameter into big expressions
    for test in tests:
        parameter_algebra_final = {}
        for lhs, rhs in test.parameter_algebra_expr.items():
            combined_expression_inputs = []
            combined_expression_children = []
            for param_expr, multiplier in rhs.items():
                assert len(multiplier) == 1, 'TODO deal with non-simple parameter algebra'
                combined_expression_inputs.append(multiplier[0])
                combined_expression_children.append(param_expr)
            combined_expression_algebra = LinearExpression(combined_expression_inputs, f'{lhs}_combiner')
            combined_expression = HeirarchicalExpression(combined_expression_algebra, combined_expression_children, lhs)
            parameter_algebra_final[lhs] = combined_expression
        test.parameter_algebra_final = parameter_algebra_final

    '''
    This has been refactored into the block above
    total_expr_inputs = []
    for name in param_names:
        old_expr = rhs[name]
        assert len(old_expr) == 1, 'TODO'
        total_expr_inputs.append(old_expr[0])
    algebra = LinearExpression(total_expr_inputs, f'{lhs_clean}_combiner')
    total_expr = HeirarchicalExpression(algebra, optional_exprs, lhs_clean)
    '''

    # we have done our job by creating test.parameter_algebra_final
    return

