from collections import defaultdict

import yaml
import os
from fixture.signals import SignalIn, SignalOut
import re

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

def dump_yaml(template, params_by_mode, mapping):
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

        # TODO for required BA mgenero needs us to treat them as a bus
        # maybe we can loop through required pins and check whether each is a bus?

        def convert_term_to_verilog(term):
            # square
            term = re.sub('I\\((.*)\\*\\*2\\)', '\\1*\\1', term)
            # cube
            term = re.sub('I\\((.*)\\*\\*3\\)', '\\1*\\1*\\1', term)
            return term

        for param, terms in params.items():
            if len(params_by_mode) == 1:
                mode_dict = {'dummy_digitalmode': 0}
            else:
                # mode dict keys are true digital pins
                names = [s.spice_name for s in template.signals if s.type_ == 'true_digital']
                mode_dict = {name:x for name,x in zip(names, binary(mode, len(names)))}
            coefs_by_mode = d.get(param, [])
            terms_verilog = {convert_term_to_verilog(k): v for k,v in terms.items()}
            coefs_this_mode = {
                    'mode': mode_dict,
                    #'coef': terms
                    'coef': terms_verilog
            }
            coefs_by_mode.append(coefs_this_mode)

            param_mapped = mapping.get(param, param)
            d[param_mapped] = coefs_by_mode

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

    def create_pin(s):
        d = {}
        # key for this whole dict is template name, value for key "name" is spice name
        d['name'] = s.spice_name
        d['description'] = f'Template: "{s.template_name}", Circuit: "{s.spice_name}"'
        # TODO inout?
        d['direction'] = 'input' if isinstance(s, SignalIn) else 'output'
        # pwl
        # logic can mean true digital or clock ...
        # vectorsize is # bits
        # Let's assume real for real and logic for digital/ba
        isreal = s.type_ in ['real', 'analog']
        d['datatype'] = 'real' if isreal else 'logic'
        d['is_optional'] = s.template_name is None

        if hasattr(s, 'value') and s.value is not None:
            d['value'] = str(s.value)

        # we treat buses as individual bits, so we never use vectorsize
        # d['vectorsize'] = 1

        return d

    pins = {}
    for s in template.signals:
        if (isinstance(s, SignalIn) or isinstance(s, SignalOut)) and s.spice_name is not None:
            # TODO I forget why we prefer template name here...
            # TODO for required BA mgenero needs us to treat them as a bus
            # maybe we can loop through required pins and check whether each is a bus?
            name = s.template_name if s.template_name is not None else s.spice_name
            pins[name] = create_pin(s)

    interface = {}
    interface['pin'] = pins
    interface['module_name'] = 'placeholder'
    interface['template_name'] = 'placeholder'
    interface['description'] = 'placeholder'
    interface['modelparam'] = collateral_dict['modelparam']

    cfg = {}
    cfg['pin'] = pins
    cfg['module_name'] = circuit.name

    interface.update(collateral_dict['interface_extras'])
    cfg.update(collateral_dict['circuit_extras'])

    interface_text = yaml.dump(interface)
    cfg_text = yaml.dump(cfg)
    return interface_text, cfg_text
    


def create_all(template, config, params):
    # TODO:
    params_text = dump_yaml(template, params, config.get('mapping', {}))
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
    
    
