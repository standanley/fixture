import sys, yaml, os
from fixture.checkpoints import Checkpoint
import fixture.config_parse as config_parse
import fault
import fixture.templates as templates
import fixture.mgenero_interface as mgenero_interface
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

    circuit_filepath = circuit_config_dict['filepath']
    simulator = Simulator(test_config_dict, circuit_filepath)


    t = TemplateClass(UserCircuit, simulator, signal_manager, extras)
    checkpoint = Checkpoint(t, 'checkpoint_folder')

    # TODO figure out UI for saving and loading
    #checkpoint.save(t, 'pickletest4.json')
    #t = checkpoint.load('pickletest4.json')

    # TODO maybe reorganize heirarchy for better checkpoints?
    #params = {} # extras?
    #for Test in t.tests:
    #    test = Test(params)
    #    tb, reads = test.create_testbench()
    #    raw_data = test.run_sim(tb, reads)
    #    data = test.analyze(raw_data)



    params_by_mode = t.go(checkpoint)

    for mode, results in params_by_mode.items():
        print('For mode', mode)
        print('category\tparam\tterm\tcoef')
        for category, result_batch in results.items():
            for param, d in result_batch.items():
                for partial_term_optional, coef in d.items():
                    print('%s\t%s\t%s\t%.3e' % (category, param, partial_term_optional, coef))

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
