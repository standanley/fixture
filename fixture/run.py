import sys, yaml, ast, os
from pathlib import Path
import fault
import fixture.templates as templates
import fixture.real_types as real_types
import fixture.sampler as sampler
import fixture.create_testbench as create_testbench
#import fixture.linearregression as lr
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

    template = getattr(templates, circuit_config_dict['template'])

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

    class UserCircuit(template):
        name = circuit_config_dict['name']
        IO = io
        extras = circuit_config_dict

        def mapping(self):
            for name, p in pins.items():
                if 'template_pin' in p:
                    setattr(self, p['template_pin'], getattr(self, name))
    vectors = sampler.Sampler.get_samples_for_circuit(UserCircuit, 50)

    tester = fault.Tester(UserCircuit)
    testbench = create_testbench.Testbench(tester)
    testbench.set_test_vectors(vectors)
    testbench.create_test_bench()

    # TODO fill in all args from SpiceTarget or remove this check
    approved_simulator_args = ['ic', 'vsup', 'bus_delim', 'ext_libs', 'inc_dirs', 'defines', 'flags']
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

    print('calling with sim dict', simulator_dict)
    print(f'Running sim, {len(vectors[0])} test vectors')
    tester.compile_and_run(test_config_dict['target'],
        simulator=test_config_dict['simulator'],
        clock_step_delay=0,
        **simulator_dict
    )
    
    print('Analyzing results')
    results = testbench.get_results()

    params_by_mode = {}
    for mode, res in enumerate(results):
        reg = Regression(UserCircuit, res)
        print(reg.results)
        params_by_mode[mode] = reg.results
    parmams_text = mgenero_interface.dump_yaml(UserCircuit, params_by_mode)
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

        mgenero_interface.create_all(UserCircuit, mgenero_params, params_by_mode)



if __name__ == '__main__':
    args = sys.argv
    circuit_config_filename = args[1]
    #test_config_filename = args[2]
    run(circuit_config_filename)

