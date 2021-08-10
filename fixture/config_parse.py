import os
import yaml
import fixture.cfg_cleaner as cfg_cleaner
from fixture.signals import create_signal, expanded
import magma
import fault
import ast
import re


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

def parse_config(circuit_config_dict):

    #cfg_cleaner.edit_cfg(circuit_config_dict)


    # load test config data
    test_config_filename = circuit_config_dict['test_config_file']
    test_config_filename_abs = path_relative(circuit_config_dict['filename'], test_config_filename)
    test_config_dict = parse_test_cfg(test_config_filename_abs)

    # Create magma pins for each pin, not signals yet
    io = []
    io_signal_info = {}
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

        # Do 2 things:
        # go through all nested buses and add flattened version to io_signal_info
        # go through all nested buses to build magma type matching heirarchy
        bus_name, name_expanded = expanded(name)
        signal_info = []
        def descend(name_or_list, indices):
            if isinstance(name_or_list, str):
                signal_info.append((name_or_list, indices))
                return dt
            child_type = None
            for i, child in enumerate(name_or_list):
                new_child_type = descend(child, indices + [i])
                if child_type is None:
                    child_type = new_child_type
                else:
                    assert child_type == new_child_type, f'Nonuniform types in array for {name}'
            return magma.Array[len(name_or_list), child_type]
        io_signal_info[name] = (bus_name, signal_info)

        dt_bus = descend(name_expanded, [])
        io += [bus_name, dt_bus]

    class UserCircuit(magma.Circuit):
        name = circuit_config_dict['name']
        IO = io

    # create spice name to template name mapping
    s2t_mapping = {}
    template_pins = circuit_config_dict['template_pins']
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
