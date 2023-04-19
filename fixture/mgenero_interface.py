from collections import defaultdict

import yaml
import os
from fixture.signals import SignalIn, SignalOut, SignalArray
from fixture.regression import Regression
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
    #return [(x//(i+1))%2 for i in range(y)]
    # x is a string tuple for some reason
    bits_str = x[1:-1].split(',')[:y]
    bits = [int(b) for b in bits_str]
    return bits

def dump_yaml(template, params_by_mode, mapping):
    d = defaultdict(list)
    for mode, params in params_by_mode.items():
        params_flat = {}
        for rhs in params.values():
            # TODO rhs.verilog(...)
            #rhs = {'gain': {list(template.signals)[0]: 42.42}}
            rhs = {'gain': 'woohoo testing raw verilog\nAnd a second line too'}
            params_flat.update(rhs)
#        # TODO I haven't found a good way to deal with required BA, so this is a bit of a hack
#        # Note all params_flat that are part of a bus
#        buss = defaultdict(list)
#        for param in params_flat:
#            if param[-1] == ']':
#                name = param.split('[')[0]
#                buss[name].append(param)
#        aggregates = defaultdict(dict)
#
#        # add buses at the end that are the sum of their components
#        # also delete entries from 'params_flat' for the individual bits
#        for name, bits in buss.items():
#            params_flat[name] = {}
#            for bit in bits:
#                terms = params_flat.pop(bit)
#                for factor, coef in terms.items():
#                    new_factor = f'{bit}*{factor}'
#                    params_flat[name][new_factor] = coef

        # TODO for required BA mgenero needs us to treat them as a bus
        # maybe we can loop through required pins and check whether each is a bus?

        def get_circuit_name(r_name):
            for s_or_a in template.signals:
                ss = s_or_a.flatten() if isinstance(s_or_a, SignalArray) else [s_or_a]
                for s in ss:
                    if Regression.regression_name(s) == r_name:
                        if s.spice_name is not None:
                            return s.spice_name
                        else:
                            return s.template_name

        def convert_term_to_verilog(term):
            if isinstance(term, SignalIn):
                term = term.spice_name
            if isinstance(term, tuple):
                return ' * '.join(convert_term_to_verilog(x) for x in term)
            if isinstance(term, SignalArray):
                assert term.bus_info['datatype'] == 'binary_analog', f'Cannot make verilog for {term} if it is not binary_analog'
                if term.bus_info['bus_type'] == 'binary_exact':
                    term_ordered = list(term) if term.bus_info['first_one'] == 'low' else list(term)[::-1]
                    # TODO I think this should be wrapped in parentheses but they break mGenero
                    return '(' + '+'.join(f'{2**i}*'+convert_term_to_verilog(b) for i, b in enumerate(term_ordered)) + ')'
                    #return '+'.join(f'{2**i}*'+convert_term_to_verilog(b) for i, b in enumerate(term_ordered))
                else:
                    assert False, 'TODO'
            if term == Regression.one_literal:
                return '1'
            # square
            term = re.sub('I\\((.*)\\*\\*2\\)', '\\1*\\1', term)
            # cube
            term = re.sub('I\\((.*)\\*\\*3\\)', '\\1*\\1*\\1', term)
            # angle braces ot brackets
            term = term.replace('<', '[').replace('>', ']')

            return term

        for param, terms in params_flat.items():
            param_mapped = mapping.get(param, param)
            if len(params_by_mode) == 1:
                mode_dict = {'dummy_digitalmode': 0}
            else:
                # mode dict keys are true digital pins
                names = [s.spice_name for s in template.signals if s.type_ == 'true_digital']
                mode_dict = {name:x for name,x in zip(names, binary(mode, len(names)))}
            coefs_by_mode = d[param_mapped]
            #terms_verilog = {convert_term_to_verilog(k): v for k,v in terms.items()}
            #coefs_this_mode = {
            #        'mode': mode_dict,
            #        #'coef': terms
            #        'coef': terms_verilog
            #}
            coefs_this_mode = {'mode': mode_dict,
                               'raw_verilog': terms}
            coefs_by_mode.append(coefs_this_mode)

    #print('hello', d['gain'][0]['mode'])

    final = {'test1': dict(d)}
    return yaml.dump(final)

def create_interface(template, collateral_dict, params):
    circuit = template.dut
    special_template_pins_rev = collateral_dict.get('pin_mapping', {})
    special_template_pins = {v: k for k, v in special_template_pins_rev.items()}

    def my_in(p, ps):
        ''' computes (p in ps) without importing mantle '''
        for p2 in ps:
            if p is p2:
                return True
        return False

    def create_pin_array(a):
        circuit_names = [x.spice_name is not None for x in s.flatten()]
        assert all(circuit_names), f'Mixed circuit/not in {s}'
        assert len(s.shape) == 1, f'mGenero buses must be 1D, {s} is {len(s.shape)}D'
        assert s.spice_name is not None, f'Expected mGenero bus to have spice bus name, {s} does not'
        # TODO should we assert something about values?

        token_s = list(a.flatten())[0]
        d = create_pin(token_s)
        d['name'] = a.spice_name
        d['description'] = f'Template: <unknown>, Circuit: "{a.spice_name}"'
        d['vectorsize'] = a.shape[0]

        # TODO I don't know whether mgenero handles values here ... it probably
        # does but you'd have to conver the bits to an integer (low order?)
        #if hasattr(s, 'value') and s.value is not None:
        #    d['value'] = str(s.value)
        return d

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
        if s.representation is not None:
            continue
        if isinstance(s, SignalArray):
            circuit_names = [x.spice_name is not None for x in s.flatten()]
            if any(circuit_names):
                # TODO I think there's a bug here with required qa. Something
                # about using the circuit bus name as the key in the next line
                # when it should be the template name - not sure exactly how it
                # should work though since I don't have any required qa models
                template_names = [x.template_name is not None for x in s.flatten()]
                if any(template_names):
                    # We want the circuit names in the verilog
                    # but the only way to communicate the template names to
                    # mgenero is one bit at a time
                    # Our (hacky) solution is to change the bus to a bunch
                    # of individual bits like in[1:0] -> in_1_, in_0_
                    # TODO what to do for a template-required bus in the model?
                    def clean_name(name):
                        # make this name friendly for verilog
                        bad_chars = '[]<>'
                        for c in bad_chars:
                            name = name.replace(c, '_')
                        return name
                    for bit in s.flatten():
                        if bit.representation is not None:
                            continue
                        # TODO doesn't work for mixed template and not in bus
                        pins[bit.template_name] = create_pin(bit)
                        pins[bit.template_name]['name'] = clean_name(bit.spice_name)
                else:
                    pins[s.spice_name] = create_pin_array(s)
        else:
            if s.spice_name is not None:
                # use the template name as the dictionary key here, and the
                # circuit name as the 'name' entry in the dict. Then mgenero
                # will do the correct translation in the verilog
                name = (s.template_name if s.template_name is not None
                        else special_template_pins.get(s.spice_name, s.spice_name))
                pins[name] = create_pin(s)
        '''
        if (isinstance(s, SignalIn) or isinstance(s, SignalOut)) and s.spice_name is not None:
            # TODO I forget why we prefer template name here...
            # TODO for required BA mgenero needs us to treat them as a bus
            # maybe we can loop through required pins and check whether each is a bus?
            name = s.template_name if s.template_name is not None else s.spice_name
            pins[name] = create_pin(s)
        '''

    interface = {}
    interface['pin'] = pins
    interface['module_name'] = 'placeholder'
    interface['template_name'] = 'placeholder'
    interface['description'] = 'placeholder'

    for param, info in collateral_dict['modelparam'].items():
        if 'get_nominal_from_signal' in info:
            signal = template.signals.from_template_name(info['get_nominal_from_signal'])
            info['value'] = sum(signal.value) / 2


    interface['modelparam'] = collateral_dict['modelparam']

    cfg = {}
    cfg['pin'] = pins
    cfg['module_name'] = circuit.name

    if 'interface_extras' in collateral_dict:

        interface.update(collateral_dict['interface_extras'])
    if 'circuit_extras' in collateral_dict:
        #cfg.update(collateral_dict['circuit_extras'])
        es = collateral_dict['circuit_extras']
        metrics = es.get('metric', {})
        for metric in list(metrics):
            if 'fixture_dependence' in metrics[metric]:
                # remove this metric if we don't have the dependence
                dependence_name = metrics[metric]['fixture_dependence']
                dependence_by_mode = [dependence_name in vs for k, vs in params.items()]
                if all(dependence_by_mode) != any(dependence_by_mode):
                    assert False, f'{dependence_name} defined for some modes but not others'
                if not all(dependence_by_mode):
                    # delete this metric since we can't model it
                    del metrics[metric]

    interface_text = yaml.dump(interface)
    cfg_text = yaml.dump(cfg)
    return interface_text, cfg_text
    


def create_all(template, config, params):
    # TODO:
    params_text = dump_yaml(template, params, config.get('mapping', {}))
    interface_text, circuit_text = create_interface(template, config, params)
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
    
    
