import fixture
dump_yaml = fixture.mgenero_interface.dump_yaml

def test_simple():
    my_d = {
        (0,):{
            'gain':{
                'constant_ones': .5,
                'I(adj)': .1
                },
            'offset':{
                'constant_ones': -.1,
                'I(adj)': 1.1
                }
            },
        (1,):{
            'gain':{
                'constant_ones': 0,
                'I(adj)': 0
                },
            'offset':{
                'constant_ones': -.1,
                'I(adj)': 1.1
                }
            }
        }
    mgenero_d = dump_yaml(my_d)
    print(mgenero_d)

if __name__ == '__main__':
    test_simple()



