import fixture.config
import types

def test_all():
    mod = fixture.config
    for k, v in mod.__dict__.items():
        if ('test' in k
                and isinstance(v, types.FunctionType)):
            print('Running test: ' + k)
            v()


if __name__ == '__main__':
    test_all()
