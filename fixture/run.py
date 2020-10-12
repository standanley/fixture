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
        tester.compile_and_run(test_config_dict['target'],
            simulator=test_config_dict['simulator'],
            clock_step_delay=0,
            tmp_dir=False,
            **simulator_dict
        )

    t = Template(UserCircuit, mapping, run_callback, extras)
    params_by_mode = t.go()

    for mode, results in params_by_mode.items():
        print('\n\nFinal results:')
        print('For mode', mode)
        print('param\tterm\tcoef')
        for param, d in results.items():
            for partial_term_optional, coef in d.items():
                print('%s\t%s\t%.3e' % (param, partial_term_optional, coef))

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

