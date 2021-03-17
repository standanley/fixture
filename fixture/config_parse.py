import os
import yaml
import fixture.cfg_cleaner as cfg_cleaner
from fixture.signals import create_signal
import magma
import fault
import ast


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

def break_bus_name2(name):
    '''
    'myname'  ->  'myname'
    'myname[0:2]<1:0>'  ->  [['myname[0]<1>', 'myname[0]<0>'], ['myname[1]<1>', ...], ...]
    :param name: name which may or may not be a bus
    :return:
    '''
    return name
    assert False, 'todo'
    return None

def parse_test_cfg(test_config_filename_abs):
    with open(test_config_filename_abs) as f:
        test_config_dict = yaml.safe_load(f)
    if 'num_cycles' not in test_config_dict and test_config_dict['target'] != 'spice':
        test_config_dict['num_cycles'] = 10**9 # default 1 second, will quit early if $finish is reached
    return test_config_dict


def parse_config(circuit_config_dict):

    cfg_cleaner.edit_cfg(circuit_config_dict)

    # load test config data
    test_config_filename = circuit_config_dict['test_config_file']
    test_config_filename_abs = path_relative(circuit_config_dict['filename'], test_config_filename)
    test_config_dict = parse_test_cfg(test_config_filename_abs)

    #Template = getattr(templates, circuit_config_dict['template'])


    # Create magma pins for each pin, not signals yet
    io = []
    pins = circuit_config_dict['pin']
    pin_params = ['datatype', 'direction', 'value']
    digital_types = ['bit', 'binary_analog', 'true_digital']
    analog_types = ['real', 'analog']
    for name, p in pins.items():
        for pin_param in p.keys():
            assert pin_param in pin_params, f'Unknown pin descriptor {pin_param} for pin {name}'
        type_string = p['datatype']
        if type_string in digital_types:
            dt = magma.Bit
        elif type_string in analog_types:
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

        value = ast.literal_eval(str(p.get('value', None)))
        io += [name, dt]

    class UserCircuit(magma.Circuit):
        name = circuit_config_dict['name']
        IO = io

    # create spice name to template name mapping
    s2t_mapping = {}
    template_pins = circuit_config_dict['template_pins']
    for template_name, spice_name in template_pins.items():
        template_name_broken = break_bus_name2(template_name)
        spice_name_broken = break_bus_name2(spice_name)
        def equate(t, s):
            err_msg = f'Mismatched bus dimensions for {template_name}, {spice_name}'
            assert type(t) == type(s), err_msg
            if type(t) == str:
                s2t_mapping[s] = t
            else:
                assert len(t) == len(s), err_msg
                for t2, s2 in zip(t, s):
                    equate(t2, s2)
        equate(template_name_broken, spice_name_broken)

    #mapping = {}
    #for name, p in pins.items():
    #    if 'template_pin' in p:
    #        if p['template_pin'] == 'ignore':
    #            i = 0
    #            while 'ignore'+str(i) in mapping:
    #                i += 1
    #            mapping['ignore'+str(i)] = name
    #        else:
    #            mapping[p['template_pin']] = name

    signals = []
    for pin_name, pin_value in pins.items():
        pin_value['spice_pin'] = getattr(UserCircuit, pin_name)
        if pin_name in s2t_mapping:
            pin_value['template_pin'] = s2t_mapping[pin_name]
        signal = create_signal(pin_value)
        signals.append(signal)


    template_class_name = circuit_config_dict['template']
    extras = circuit_config_dict['extras']
    return (UserCircuit, template_class_name, signals, test_config_dict, extras)
