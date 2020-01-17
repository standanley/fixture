from fixture import template_master

'''
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
'''

def extract_pzs(nps, nzs, x, y):
    # TODO
    return ([42.42]*nps, [42.42]*nzs)


def dynamic(template):
    print('IN DYNAMIC WRAPPER', template, template==template_master)
    # NOTE: the only reason I inherit directly from TemplateMaster
    # here is because I check whether a class is a template by checking
    # whether it's a direct subclass of TemplateMaster
    class Dynamic(template, template_master.TemplateMaster):
        dynamic_reads = {}

        # create function for reading transient in run_single_test
        @classmethod
        def read_transient(self, tester, port, duration):
            r = tester.read(port, style='block', params={'duration':duration})
            self.dynamic_reads[port] = r

        # wrap run_single_test to return read_transient
        @classmethod
        def run_single_test(self, *args, **kwargs):
            #print('In the wrapper for run now! args are:', args)
            #ret = template.run_single_test(*args, **kwargs)
            ret = super().run_single_test(*args, **kwargs)
            err = ('If you use the Dynamic Template type, you must call '
                'read_transient in your run_single_test!')
            assert len(self.dynamic_reads) > 0, err
            dynamic_reads = self.dynamic_reads
            self.latest_dynamic_reads = {}
            return (ret, dynamic_reads)

        # wrap process_single_test to process the block read
        @classmethod
        def process_single_test(self, reads, *args, **kwargs):
            # we purposely put dynamic_reads in a scope the template creator
            # can access so that they can edit it before we process
            reads_orig, block_reads = reads
            self.dynamic_reads = {p:r.value for p,r in block_reads.items()}
            ret_dict = super().process_single_test(reads_orig, *args, **kwargs)
            for port, (x,y) in self.dynamic_reads.items():
                ps, zs = extract_pzs(1, 0, x, y)
                p1 = ps[0]

                # add these ps zs to parameter algebra if they are not already there
                name = f'{self.get_name(port)}_p1'
                if not name in [n for n,_ in self.parameter_algebra]:
                    self.parameter_algebra.append((name, {name:'1'}))

                # add to the dict so they can be used for regression later
                ret_dict[name] = p1
            return ret_dict

    return Dynamic







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

