from fixture import template_master
#from fixture.templates import SimpleAmpTemplate
#from . import templates

#test = fixture.templates.SimpleAmpTemplate
import fixture
#fixture.SimpleAmpTemplate
#fixture.templates
#print(fixture.__dict__)
from . import templates
#templates.simple_amp
#templates.SimpleAmpTemplate
#from . import SimpleAmpTemplate
templates.PhaseBlenderTemplate


def extract_pzs(nps, nzs, x, y):
    # TODO
    return ([42.42]*nps, [42.42]*nzs)


class Dynamic(template_master.TemplateKind):
    def __new__(metacls, name, bases, dct):
        cls = super(Dynamic, metacls).__new__(metacls, name, bases, dct)
        print('\ntop of new', cls, hasattr(cls, 'in_single'))

        if hasattr(cls, 'latest_dynamic_read') or cls == templates.simple_amp.SimpleAmpTemplate:
            print('Skipping dynamic wrapper for', cls)
            return cls

        print(cls.parameter_algebra)
        #print('pa was', cls.parameter_algebra)

        # add pzs to parameter_algebra
        cls.parameter_algebra += ('p1', {'p1': '1'})

        # create function for reading transient in run_single_test
        cls.latest_dynamic_read = None
        def read_transient(self, tester, duration):
            r = tester.read(self.out_single, style='block', params={'duration':duration})
            self.latest_dynamic_read = r
        cls.read_transient = read_transient

        # wrap run_single_test to return read_transient
        def dec_run(f):
            def wrapper(*args, **kwargs):
                print('In the wrapper for run now! args are:', args)
                ret = f(*args, **kwargs)
                err = ('If you use the Dynamic Template type, you must call '
                    'read_transient in your run_single_test!')
                assert cls.latest_dynamic_read != None, err
                block_read = cls.latest_dynamic_read
                cls.latest_dynamic_read = None
                return (ret, block_read)
            return wrapper
        print('Wrapping run right now!', cls)
        cls.run_single_test = dec_run(cls.run_single_test)

        # wrap process_single_test to process the block read
        def dec_process(f):
            def wrapper(self, reads, **kwargs):
                reads_orig, block_read = reads
                ret_dict = f(self, reads_orig, **kwargs)
                x, y = block_read.value
                ps, zs = extract_pzs(1, 0, x, y)
                p1 = ps[0]
                ret_dict['p1'] = p1
                return ret_dict
            return wrapper
        cls.process_single_test = dec_process(cls.process_single_test)

        return cls







def poke_binary_analog(tester, port, value):
    '''
    Assuming port is binary analog with port[0] as the
    low-order bit, poke "value" assuming value is scaled
    0 to 1.
    Note that if value is random, this just assigns a random value to each
    port independently, which is what we want to do for testing. Actually
    using "value" is helpful if the port happens to be binary-weighted
    and the value chosen with LHS or orthogonal sampling
    '''

    bits = len(port)
    int_value = int(value * 2**bits)
    for i in range(bits):
        tester.poke(port[i], int_value & 1)
        int_value >>= 1

