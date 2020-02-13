import yaml

def dummy_dump(params, filename):
    #out = {'a': ['b', 'c', {'d':'e', 'f':'g'}]}
    for k in params:
        params[k][0]['mode']={'dummy_digitalmode':0}
    out = {'test1':params}
    with open(filename, 'w') as f:
        f.write(yaml.dump(out, default_flow_style=False))

if __name__=='__main__':
    dummy_dump({'my_param':[{'coef':{'offset':5.6}}]}, 'test.yaml')

