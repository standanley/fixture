import sys, yaml, ast
from pathlib import Path
import fault
import fixture.templates as templates
import fixture.real_types as real_types
import fixture.sampler as sampler
import fixture.create_testbench as create_testbench
import fixture.linearregression as lr

def run(circuit_config_filename):
    with open(circuit_config_filename) as f:
        circuit_config_dict = yaml.safe_load(f)
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
    ins, outs = results
    results_reformatted = [ins[0], outs[0]]

    iv_names, dv_names = testbench.get_input_output_names()
    #formula = {'out':'in_ + I(in_**2) + I(in_**3)'}
    regression = lr.LinearRegressionSM(iv_names, dv_names, results_reformatted)
    regression.run()
    print(regression.get_summary()[dv_names[0]])

    


if __name__ == '__main__':
    args = sys.argv
    circuit_config_filename = args[1]
    run(circuit_config_filename)

