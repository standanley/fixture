from collections import defaultdict

import yaml
import os
from magma import Array
import fixture.real_types as rt

'''
A parameter file for mgenero looks like this:

{for test:
    { for dv: # an example of dv is 'dcgain' or 'fz1'
        [for true digital modes:
            {'mode': 'modename' # 'dummy_digitalmode' is common
             'coef': { for term:
                 'term_name' : value # term name might be 'offset'
             }
            }
        ]
    }
}
'''


def binary(x, y):
    return [(x//(i+1))%2 for i in range(y)]

def dump_yaml(dut, params_by_mode):
    d = {}
    for mode, params in params_by_mode.items():
#        # TODO I haven't found a good way to deal with required BA, so this is a bit of a hack
#        # Note all params that are part of a bus
#        buss = defaultdict(list)
#        for param in params:
#            if param[-1] == ']':
#                name = param.split('[')[0]
#                buss[name].append(param)
#        aggregates = defaultdict(dict)
#
#        # add buses at the end that are the sum of their components
#        # also delete entries from 'params' for the individual bits
#        for name, bits in buss.items():
#            params[name] = {}
#            for bit in bits:
#                terms = params.pop(bit)
#                for factor, coef in terms.items():
#                    new_factor = f'{bit}*{factor}'
#                    params[name][new_factor] = coef


        for param, terms in params.items():
            if len(params_by_mode) == 1:
                mode_dict = {'dummy_digitalmode': 0}
            else:
                # mode dict keys are true digital pins
                names = [dut.get_name_circuit(p) for p in dut.inputs_true_digital]
                mode_dict = {name:x for name,x in zip(names, binary(mode, len(names)))}
            coefs_by_mode = d.get(param, [])
            coefs_this_mode = {
                    'mode': mode_dict,
                    'coef': terms
                }
            coefs_by_mode.append(coefs_this_mode)
            d[param] = coefs_by_mode

    #print('hello', d['gain'][0]['mode'])

    final = {'test1':d}
    return yaml.dump(final)

def create_interface(template, collateral_dict):
    circuit = template.dut


    def my_in(p, ps):
        ''' computes (p in ps) without importing mantle '''
        for p2 in ps:
            if p is p2:
                return True
        return False

    def create_pin(p, spice_name):
        d = {}
        # key for this whole dict is template name, value for key "name" is spice name
        d['name'] = spice_name
        d['description'] = str(p)
        # TODO inout?
        d['direction'] = 'input' if not p.isinput() else 'output'
        # pwl
        # logic can mean true digital or clock ...
        # vectorsize is # bits
        # Let's assume real for real and logic for digital/ba
        isreal = rt.is_real(p)
        d['datatype'] = 'real' if isreal else 'logic'
        d['is_optional'] = my_in(p,
            (template.inputs_analog + template.inputs_ba + template.inputs_pinned))

        # TODO array of reals?
        if isinstance(p, Array):
            d['vectorsize'] = len(p)

        return d

    ## mapping from fixture template name to spice name
    #pin_name_mapping = {}
    #for template_port_name in circuit.required_ports:
    #    template_port = getattr(circuit, template_port_name)
    #    # TODO use the mgenero template name for the key
    #    pin_name_mapping[template_port] = template_port.fixture_name

    pins = {}
    for p_name, _ in circuit.IO.items():
        p = getattr(circuit, p_name)
        spice_name = template.get_name_circuit(p)
        template_name = template.get_name_template(p)
        # this next line key should not be spice name
        pins[template_name] = create_pin(p, spice_name)

    interface = {}
    interface['pin'] = pins
    interface['module_name'] = 'placeholder'
    interface['template_name'] = 'placeholder'
    interface['description'] = 'placeholder'
    interface['modelparam'] = collateral_dict['modelparam']

    cfg = {}
    cfg['pin'] = pins
    cfg['module_name'] = circuit.name

    interface_text = yaml.dump(interface)
    cfg_text = yaml.dump(cfg)
    return interface_text, cfg_text
    


def create_all(template, config, params):
    # TODO:
    params_text = dump_yaml(template, params)
    interface_text, circuit_text = create_interface(template, config)
    generate_text = get_generate_text(config['template_name'])
    directory = config['build_folder']

    with open(os.path.join(directory, 'params.yaml'), 'w') as f:
        f.write(params_text)
    with open(os.path.join(directory, 'circuit.cfg'), 'w') as f:
        f.write(circuit_text)
    with open(os.path.join(directory, 'interface.yaml'), 'w') as f:
        f.write(interface_text)
    with open(os.path.join(directory, 'generate.py'), 'w') as f:
        f.write(generate_text)

def get_generate_text(template_name):
    text = f''' # autogenerated
from dave.mgenero.mgenero import ModelCreator
config = 'circuit.cfg'
interface = 'interface.yaml'
template = '{template_name}'
params = 'params.yaml'
intermediate = 'template.intermediate.sv'
output = 'final.sv'

m = ModelCreator(config, interface)
m.generate_model(template, intermediate)
m.backannotate_model(intermediate, output, params)
    '''
    return text
    
    
