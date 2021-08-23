import os
import yaml
import fixture.cfg_cleaner as cfg_cleaner
from fixture.signals import create_signal, expanded, parse_bus, SignalArray
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





    # go through pin definitions and create list of
    # [pin_info, circuit_name, template_name]
    # We don't create signals yet because we don't have template names and
    # create_signal has some checks involving that
    # temporarily use magma type (disregard bus?) in place of spice pin
    io = []
    #io_signal_info = {}
    signals_by_name = {}
    pins = circuit_config_dict['pin']
    pin_params = ['datatype', 'direction', 'value', 'electricaltype']
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

        bus_name, indices_parsed, info_parsed, bits = parse_bus(name)
        if len(indices_parsed) == 0:
            # dict, cname, tname
            signals_by_name[name] = [p, name, None]
        else:
            # array of (dict, cname, tname)
            indices_limits = [max(i_range)+1 for i_range in indices_parsed]
            if bus_name in signals_by_name:
                assert False, 'TODO'
            else:
                a = np.zeros(indices_limits, dtype=object)
                for bit_name, location in bits:
                    a[location] = [p, bit_name, None]
                signals_by_name[bus_name] = a

        # At this point, there are 2 problems with the signals:
        # 1) the spice_pin entry is just a magma datatype, not the object
        # 2) the template_name entry is always None

    # parse template to pin mapping - replace None with template name in
    t2s_mapping = {}

    '''
    issue right now: 
    t: c
    t: c[0:7]
    t: c[7:0]
    t[0:7]: c
    t[7:0]: c
    t[7:0]: c[0:7]
    In which case(s) does t[0] map to c[0]?
    I think all cases except the last one
    '''


    template_pins = circuit_config_dict['template_pins']
    for t, c in template_pins.items():
        t_bus_name, t_indices, _, _ = parse_bus(t)
        c_bus_name, c_indices, _, _ = parse_bus(c)
        c_array = signals_by_name[c_bus_name]

        c_shape = getattr(c_array, 'shape', [])
        assert len(c_shape) >= len(c_indices), f'Too many indices in {c}'
        # extend c_indices so it goes all the way to the end of c
        c_indices += [(0, x-1) for x in c_shape[len(c_indices):]]

        # if the template array is 1 entry, that entry encompases the whole circuit array
        # if the template array is multiple entries, they must match with the circuit
        #    entries 1 to 1 until the template entries run out of dimensions
        #    if circuit runs out of dimensions first, that's an error
        # Dimensions that aren't a range are kinda skipped in this mapping
        def match(t_indices_used, t_indices, c_a, c_indices):
            #print('match called with', t_indices_used, t_indices, getattr(c_a, 'shape', []), c_indices)
            if len(c_indices) == 0:
                # c is out of indices but t is not - that's an error
                assert len(t_indices) == 0, f'error mapping {t} to {c}, too many dims in {c}'
                # we should be at the end of the array
                assert not isinstance(c_a, np.ndarray), 'internal error in config_parse?'
                print('mapping', t_indices_used, c_a)
            elif len(c_indices[0]) == 1:
                # not a range for c, so descend on c but don't move on t
                match(t_indices_used,
                      t_indices,
                      c_a[c_indices[0][0]],
                      c_indices[1:])
            elif len(t_indices) == 0:
                # t_indices_used gets mapped to c_array[c_indices]
                assert len(c_indices[0]) == 2, 'internal error in config_parse?'
                # Add a dimension to t so it matches c
                match(t_indices_used,
                      t_indices + [c_indices[0]],
                      c_a,
                      c_indices)
            elif len(t_indices[0]) == 1:
                # not a range for t, so keep it in the list but don't move on c
                match(t_indices_used + [t_indices[0][0]],
                      t_indices[1:, ],
                      c_a,
                      c_indices)
            else:
                # they both have ranges; they must match
                tr = range_inclusive(*t_indices[0])
                cr = range_inclusive(*c_indices[0])
                assert len(tr) == len(cr), f'Mismatched shapes for {t}, {c}'
                for ti, ci in zip(tr, cr):
                    match(t_indices_used + [ti],
                          t_indices[1:],
                          c_a[ci],
                          c_indices[1:])
        match([], t_indices, c_array, c_indices)


        # first, get the spice bits that this actually corresponds to
        #s_indices = [si for _, si in s_bits]
        #assert s_bus_name in signals_by_name, f'Unknown circuit pin {s_bus_name} in template pin mapping'


    class UserCircuit(magma.Circuit):
        name = circuit_config_dict['name']
        IO = io

    # create spice name to template name mapping
    s2t_mapping = {}
    for template_name, spice_name in template_pins.items():
        _, template_name_expanded = expanded(template_name)
        _, spice_name_expanded = expanded(spice_name)
        def equate(t, s):
            err_msg = f'Mismatched bus dimensions for {template_name}, {spice_name}'
            assert type(t) == type(s), err_msg
            if type(t) == str:
                s2t_mapping[s] = t
            else:
                assert len(t) == len(s), err_msg
                for t2, s2 in zip(t, s):
                    equate(t2, s2)
        equate(template_name_expanded, spice_name_expanded)

    signals = []
    for pin_name, pin_value in pins.items():
        magma_name, components = io_signal_info[pin_name]
        value = ast.literal_eval(str(pin_value.get('value', None)))
        pin_value['value'] = value

        for component, indices in components:
            pin_value_component = pin_value.copy()
            magma_obj = getattr(UserCircuit, magma_name)
            for i in indices:
                magma_obj = magma_obj[i]

            pin_value_component['spice_pin'] = magma_obj
            pin_value_component['spice_name'] = component
            if component in s2t_mapping:
                pin_value_component['template_pin'] = s2t_mapping.pop(component)

            signal = create_signal(pin_value_component)
            signals.append(signal)
    assert len(s2t_mapping) == 0, f'Unrecognized spice pin "{list(s2t_mapping)[0]}" in template_pin mapping'

    template_class_name = circuit_config_dict['template']
    extras = parse_extras(circuit_config_dict['extras'])
    return UserCircuit, template_class_name, signals, test_config_dict, extras
