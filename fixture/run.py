import sys, yaml, os
import fixture.config_parse as config_parse
from pathlib import Path
import fault
import fixture.templates as templates
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

    UserCircuit, template_name, signal_manager, test_config_dict, extras = config_parse.parse_config(circuit_config_dict)
    tester = fault.Tester(UserCircuit)
    TemplateClass = getattr(templates, template_name)


    # TODO fill in all args from SpiceTarget or remove this check
    approved_simulator_args = ['ic', 'vsup', 'bus_delim', 'ext_libs', 'inc_dirs',
                               'defines', 'flags', 't_step', 'num_cycles',
                               'conn_order', 'no_run', 'directory', 't_tr',
                               'dump_waveforms', 'timescale']
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
        no_run_dict = {}
        if no_run:
            print('SKIPPING SIMULATION, using results from last time')
            # I pass it in this dict because no_run=False doesn't work for all simulators
            no_run_dict['no_run'] = False

        tester.compile_and_run(test_config_dict['target'],
            simulator=test_config_dict['simulator'],
            clock_step_delay=0,
            tmp_dir=False,
            **no_run_dict,
            **simulator_dict
        )

    t = TemplateClass(UserCircuit, run_callback, signal_manager, extras)
    params_by_mode = t.go()

    for mode, results in params_by_mode.items():
        print('For mode', mode)
        print('param\tterm\tcoef')
        for param, d in results.items():
            for partial_term_optional, coef in d.items():
                print('%s\t%s\t%.3e' % (param, partial_term_optional, coef))

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
    run(circuit_config_filename)
