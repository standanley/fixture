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

    ##UserCircuit, template_name, signal_manager, test_config_dict, optional_input_info, extras = config_parse.parse_config(circuit_config_dict)
    #tester = fault.Tester(UserCircuit)
    #TemplateClass = getattr(templates, template_name)

    #circuit_filepath = circuit_config_dict['filepath']
    #assert os.path.exists(circuit_filepath), f'Circuit filepath "{circuit_filepath}" not found'
    #simulator = Simulator(test_config_dict, circuit_filepath)


    #t = TemplateClass(UserCircuit, simulator, signal_manager, extras)



    t = config_parse.parse_config(circuit_config_dict)
    #TODO this might be a good place to build up nonlinear expressions
    #config_parse.parse_optional_input_info(circuit_config_dict, t.tests)

    checkpoint = Checkpoint(t, 'checkpoint_folder')

    # TODO figure out UI for saving and loading
    #checkpoint.save(t, 'pickletest4.json')
    #t = checkpoint.load('pickletest4.json')

    # TODO maybe reorganize hierarchy for better checkpoints?
    #params = {} # extras?
    #for Test in t.tests:
    #    test = Test(params)
    #    tb, reads = test.create_testbench()
    #    raw_data = test.run_sim(tb, reads)
    #    data = test.analyze(raw_data)



    all_checkpoints = {
        'choose_inputs': True,
        'run_sim': True,
        'run_analysis': True,
        'run_post_process': True,
        'run_regression': True,
    }

    # TODO move this block to config_parse
    checkpoint_controller = {}
    if 'checkpoint_controller' in circuit_config_dict:
        cc_str = circuit_config_dict['checkpoint_controller']
        test_mapping = {str(test): test for test in t.tests}
        for test_str, val in cc_str.items():
            assert test_str in test_mapping, f'Unknown test "{test_str}" in checkpoint controller'
            test = test_mapping[test_str]
            if isinstance(val, bool):
                if val:
                    checkpoint_controller[test] = all_checkpoints
            elif isinstance(val, dict):
                assert set(val) == set(all_checkpoints), f'If specifying checkpoints for a test, must specify all keys. You gave {set(val)}, should be {set(all_checkpoints)}'
                checkpoint_controller[test] = val
            else:
                assert False, f'Confused by type of "{val}" in checkpoint controller for test "{test_str}"'
    else:
        print(f'No tests specified, defaulting to all tests: {[str(test) for test in t.tests]}')
        for test in t.tests:
            checkpoint_controller[test] = all_checkpoints

    params_by_mode = t.go(checkpoint, checkpoint_controller)

    # No need to print like this now because it prints the verilog representation
    #for mode, results in params_by_mode.items():
    #    print('For mode', mode)
    #    print('category\tparam\tterm\tcoef')
    #    for category, result_batch in results.items():
    #        for param, d in result_batch.items():
    #            for partial_term_optional, coef in d.items():
    #                print('%s\t%s\t%s\t%.3e' % (category, param, partial_term_optional, coef))
    for test in checkpoint_controller.keys():
        print(f'Results for {test}:')
        for lhs, rhs in test.parameter_algebra_final.items():
            print(f'\tModel for {lhs}:')
            coef_names = [f'c{i}' for i in range(rhs.NUM_COEFFICIENTS)]
            for line in rhs.verilog(lhs, coef_names):
                print(f'\t\t{line}')
            for name, val in zip(coef_names, rhs.x_opt):
                print(f'\t\t{name} = {val};')
            print()

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
