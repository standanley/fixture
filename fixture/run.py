import sys, yaml, ast, os
from pathlib import Path
import magma
import fault
import fixture.templates as templates
import fixture.real_types as real_types
import fixture.sampler as sampler
import fixture.create_testbench as create_testbench
from fixture import Regression
import fixture.mgenero_interface as mgenero_interface
import fixture.cfg_cleaner as cfg_cleaner
from fixture.signals import create_signal

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

def edit_paths(config_dict, config_filename, params):
    for param in params:
        if param in config_dict:
            old = config_dict[param]
            new = path_relative(config_filename, old)
            config_dict[param] = new

def run(circuit_config_filename):
    with open(circuit_config_filename) as f:
        circuit_config_dict = yaml.safe_load(f)
        circuit_config_dict['filename'] = circuit_config_filename
    edit_paths(circuit_config_dict, circuit_config_filename, ['filepath', 'mgenero'])
    _run(circuit_config_dict)

def _run(circuit_config_dict):
    cfg_cleaner.edit_cfg(circuit_config_dict)

    # load test config data
    test_config_filename = circuit_config_dict['test_config_file']
    test_config_filename_abs = path_relative(circuit_config_dict['filename'], test_config_filename)
    with open(test_config_filename_abs) as f:
        test_config_dict = yaml.safe_load(f)
    if 'num_cycles' not in test_config_dict and test_config_dict['target'] != 'spice':
        test_config_dict['num_cycles'] = 10**9 # default 1 second, will quit early if $finish is reached

    Template = getattr(templates, circuit_config_dict['template'])

    # generate IO
    io = []
    pins = circuit_config_dict['pin']
    for name, p in pins.items():
        dt = getattr(real_types, p['datatype'])
        value = ast.literal_eval(str(p.get('value', None)))
        dt = dt(value)
        direction = getattr(real_types, p['direction'])
        dt = direction(dt)
        if 'width' in p:
            dt = real_types.Array(p['width'], dt)
        io += [name, dt]

    class UserCircuit(magma.Circuit):
        name = circuit_config_dict['name']
        IO = io

    mapping = {}
    for name, p in pins.items():
        if 'template_pin' in p:
            if p['template_pin'] == 'ignore':
                i = 0
                while 'ignore'+str(i) in mapping:
                    i += 1
                mapping['ignore'+str(i)] = name
            else:
                mapping[p['template_pin']] = name

    signals = []
    for pin_name, pin_value in pins.items():
        pin_value['spice_name'] = pin_name
        signal = create_signal(pin_value)
        signals.append(signal)
        pass


    extras = circuit_config_dict

    tester = fault.Tester(UserCircuit)


    # TODO fill in all args from SpiceTarget or remove this check
    approved_simulator_args = ['ic', 'vsup', 'bus_delim', 'ext_libs', 'inc_dirs',
                               'defines', 'flags', 't_step', 'num_cycles',
                               'conn_order', 'no_run', 'directory', 't_tr']
    simulator_dict = {k:v for k,v in test_config_dict.items() if k in approved_simulator_args}

    # make sure to put the circuit file location in the right arg
    if test_config_dict['target'] == 'spice':
        model_path_key = 'model_paths'
    else:
        model_path_key = 'ext_libs'
        simulator_dict['ext_model_file'] = True
    mps = simulator_dict.get(model_path_key, [])
    mps.append(Path(circuit_config_dict['filepath']).resolve())
    simulator_dict[model_path_key] = mps

    # flgs will later get shell escaped, but I think the user should have escaped them already
    # ran into problems when a flag was like '-define NCVLOG'
    #if 'flags' in simulator_dict:
    #    flags = [x for f in simulator_dict['flags'] for x in f.split()]
    #    simulator_dict['flags'] = flags

    def run_callback(tester):
        print('calling with sim dict', simulator_dict)
        #simulator_dict['directory'] = f'build_{name}'

        no_run = False
        if no_run:
            print('SKIPPING SIMULATION, using results from last time')

        tester.compile_and_run(test_config_dict['target'],
            simulator=test_config_dict['simulator'],
            clock_step_delay=0,
            tmp_dir=False,
            no_run=no_run,
            **simulator_dict
        )

    t = Template(UserCircuit, mapping, run_callback, extras, signals)
    params_by_mode = t.go()

    for mode, results in params_by_mode.items():
        print('For mode', mode)
        print('param\tterm\tcoef')
        for param, d in results.items():
            for partial_term_optional, coef in d.items():
                #temp = coef * 225/1.2
                print('%s\t%s\t%.3e' % (param, partial_term_optional, coef))

    # TEST for differential stuff
    #import numpy as np
    #a, b, c, d = [params_by_mode[0][x]['1'] for x in ['A', 'B', 'C', 'D']]
    #abcd = np.array([[a, b], [c, d]])
    #m = np.array([[1, -1], [.5, .5]])
    #minv = np.linalg.inv(m)
    #[[w, x], [y, z]] = m @ abcd @ minv
    #print('w', w)
    #print('x', x)
    #print('y', y)
    #print('z', z)
    #w1, x1, y1, z1 = [params_by_mode[0][x]['1'] for x in ['gain', 'gain_from_cm', 'gain_to_cm', 'cm_gain']]
    #print('w1', w1)
    #print('x1', x1)
    #print('y1', y1)
    #print('z1', z1)


    #if DEBUG:
    #    vals = {k:v.value for k,v in DEBUG_DICT.items()}
    #    pass
    #    import matplotlib.pyplot as plt
    #    leg = []
    #    for k,v in vals.items():
    #        plt.plot(v[0], v[1])
    #        leg.append(k)
    #    plt.legend(leg)
    #    plt.show()

    if 'mgenero' in circuit_config_dict:
        mgenero_config_dir = circuit_config_dict['mgenero']
        with open(mgenero_config_dir) as f:
            mgenero_params = yaml.safe_load(f)

        # make sure the build folder directory is absolute
        dir_unclean = mgenero_params['build_folder']
        dir_clean = path_relative(mgenero_config_dir, dir_unclean)
        mgenero_params['build_folder'] = dir_clean

        # create build folder
        if not os.path.exists(dir_clean):
            os.makedirs(dir_clean)

        mgenero_interface.create_all(t, mgenero_params, params_by_mode)



if __name__ == '__main__':
    args = sys.argv
    circuit_config_filename = args[1]
    #test_config_filename = args[2]
    run(circuit_config_filename)

