import yaml

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


def dump_yaml(params_by_mode):
    d = {}
    for mode, params in params_by_mode.items():
        for param, terms in params.items():
            mode_dict = {'true_digital_%d'%i:x for i,x in enumerate(mode)}
            coefs_by_mode = d.get(param, [])
            coefs_this_mode = {
                    'mode': mode_dict,
                    'coef': terms
                }
            coefs_by_mode.append(coefs_this_mode)
            d[param] = coefs_by_mode

    print('hello', d['gain'][0]['mode'])

    final = {'test1':d}
    return yaml.dump(final)
