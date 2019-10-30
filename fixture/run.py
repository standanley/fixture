import sys, yaml, ast, os
from pathlib import Path
import fault
import fixture.templates as templates
import fixture.real_types as real_types
import fixture.sampler as sampler
import fixture.create_testbench as create_testbench
import fixture.linearregression as lr

def path_relative(path_to_config, path_from_config):
    ''' Interpret path names specified in config file
    We want path names relative to the current directory (or absolute).
    But we assume relative paths in the config mean relative to the config.
    '''
    if os.path.isabs(path_from_config):
        return path_from_config
    folder = os.path.dirname(path_to_config)
    res = os.path.join(folder, path_from_config)
    print(res)
    return res

def edit_paths(config_dict, config_filename, params):
    for param in params:
        old = config_dict[param]
        new = path_relative(config_filename, old)
        config_dict[param] = new
        print('changed path', old, 'to', new)

def run(circuit_config_filename):
    with open(circuit_config_filename) as f:
        circuit_config_dict = yaml.safe_load(f)
    edit_paths(circuit_config_dict, circuit_config_filename, ['filepath'])
    #for p in circuit_config_dict['pin']:
    #    print(p)
    #    print(circuit_config_dict['pin'][p])
    _run(circuit_config_dict)

def _run(circuit_config_dict):
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

        def mapping(self):
            for name, p in pins.items():
                if 'template_pin' in p:
                    setattr(self, p['template_pin'], getattr(self, name))

    vectors = sampler.Sampler.get_samples_for_circuit(UserCircuit, 5)
    print(vectors)

    tester = fault.Tester(UserCircuit)
    testbench = create_testbench.Testbench(tester)
    testbench.set_test_vectors(vectors)
    testbench.create_test_bench()

    print(f'Running sim, {len(vectors[0])} test vectors')
    tester.compile_and_run('spice',
        simulator='ngspice',
        model_paths = [Path(circuit_config_dict['filepath']).resolve()]
    )

    print('Analyzing results')
    results = testbench.get_results()
    results_reformatted = results[0]

    iv_names, dv_names = testbench.get_input_output_names()
    #formula = {'out':'in_ + I(in_**2) + I(in_**3)'}
    regression = lr.LinearRegressionSM(iv_names, dv_names, results_reformatted)
    regression.run()
    print(regression.get_summary()[dv_names[0]])

    


if __name__ == '__main__':
    args = sys.argv
    circuit_config_filename = args[1]
    run(circuit_config_filename)

