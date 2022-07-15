from pathlib import Path

class Simulator:
    def __init__(self, test_config_dict, circuit_filepath):


        # TODO fill in all args from SpiceTarget or remove this check
        approved_simulator_args = ['ic', 'vsup', 'bus_delim', 'ext_libs', 'inc_dirs',
                                   'defines', 'flags', 't_step', 'num_cycles',
                                   'conn_order', 'no_run', 'directory', 't_tr',
                                   'dump_waveforms', 'timescale', 'pwl_signals', 'rz',
                                   'sim_cmd']
        simulator_dict = {k: v for k, v in test_config_dict.items() if
                          k in approved_simulator_args}

        # make sure to put the circuit file location in the right arg
        if test_config_dict['target'] == 'spice':
            model_path_key = 'model_paths'
        else:
            model_path_key = 'ext_libs'
            simulator_dict['ext_model_file'] = True
        mps = simulator_dict.get(model_path_key, [])
        mps.append(Path(circuit_filepath).resolve())
        simulator_dict[model_path_key] = mps

        self.simulator_dict = simulator_dict
        self.test_config_dict = test_config_dict


    def run(self, tester, run_dir=None, no_run=False):
        # flgs will later get shell escaped, but I think the user should have escaped them already
        # ran into problems when a flag was like '-define NCVLOG'
        # if 'flags' in simulator_dict:
        #    flags = [x for f in simulator_dict['flags'] for x in f.split()]
        #    simulator_dict['flags'] = flags

        print('calling with sim dict', self.simulator_dict)
        # simulator_dict['directory'] = f'build_{name}'

        no_run_dict = {}
        if no_run:
            print('SKIPPING SIMULATION, using results from last time')
            # I pass it in this dict because no_run=False doesn't work for all simulators
            no_run_dict['no_run'] = True

        res = tester.compile_and_run(self.test_config_dict['target'],
                               simulator=self.test_config_dict['simulator'],
                               clock_step_delay=0,
                               tmp_dir=run_dir is None,
                               directory=run_dir,
                               **no_run_dict,
                               **self.simulator_dict
                               )