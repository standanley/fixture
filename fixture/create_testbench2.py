import fixture
import magma
from fixture import real_types
import re
import ast
import yaml

class SignalIn():
    def __init__(self,
                 value,
                 type_,
                 get_random,
                 auto_set,
                 spice_name,
                 template_name#,
                 #bus_name,
                 #bus_i
            ):
        # TODO: do we need bus info?
        self.value = value
        self.type_ = type_
        self.get_random = get_random
        self.auto_set = auto_set
        self.spice_name = spice_name
        self.template_name = template_name
        #self.bus_name = bus_name
        #self.bus_i = bus_i


class EmptyTemplate():
    def __init__(self, config_data, dut):
        # Make sure template writer is doing things correctly
        assert hasattr(self, 'tests'), 'Template writer must designate tests'
        self.Test.template = self

        # Now actual work that every template does
        self.signals_in = []
        # NOTE each instance of test will make its own copy on instantiation
        self.Test.signals_in = self.signals_in
        self.buses_in = {}
        self.spice = self.Mapping(dut)
        self.Test.spice = self.spice
        self.parse_spice_info(config_data['spice_io'])


    class Mapping():
        def __init__(self, dut):
            self.dut = dut
            self.mapping = {}

        def map(self, spice_name, template_name):
            # recall that spice buses will always be expanded at this point

            def assign_template(name, val):
                '''
                return "val", but if "name" is an entry in a bus, have "val"
                packed appropriately into a list (or nested list). "existing"
                is either nothing or a partially-populated list to be used
                '''
                #m = re.match('(.*)\\[[(0-9)]+\\](.*)', template_name)
                indices = [g[1:-1] for g in re.findall('<[0-9]+>', name)]
                name = re.match('^(.*?)(<[0-9]+>)*$', name).group(1)

                def rec(x, indices):
                    if len(indices)==0:
                        return val
                    i = indices[0]
                    if x is None:
                        x = []
                    assert type(x) == list, f'Cannot overwrite value {x} with remaining indices {name}{indices}={val}'
                    if len(x) < i+1:
                        x += [None] * (i+1-len(x))
                    x[i] = rec(x[i], indices[1:])
                    return x

                x = self.mapping.get(name, None)
                v = rec(x, indices)
                self.mapping[name] = v

            spice = getattr(self.dut, spice_name)
            assign_template(template_name, spice)


        def __getattr__(self, item):
            if item in self.mapping:
                return self.mapping[item]
            else:
                super()
                #assert False, f'No required port named "{item}"'


    def parse_spice_info(self, pin_info):
        for name in pin_info:
            d = pin_info[name]
            d['spice_name'] = name
            s = self.create_signal_in(d)
            self.signals_in.append(s)

    #def create_signal_in(
    #        self,
    #        value = None,
    #        type_ = 'analog',
    #        auto_set=None,
    #        spice_pin=None,
    #        template_name=None,
    #        bus_name=None,
    #        bus_i=None,
    #        width=None):
    def create_signal_in(self, pin_dict):
        value = pin_dict.get('value', None)
        type_ = pin_dict.get('type', 'analog')
        get_random = pin_dict.get('get_random',
            (type(value) == tuple) or (type_ == 'binary_analog' and value == None))
        auto_set = pin_dict.get('auto_set',
            get_random or type(value) == int or type(value) == float)
        spice_name = pin_dict.get('spice_name', None)
        template_name = pin_dict.get('template_name', None)

        # now think about buses
        # TODO right now we have no way of knowing whether a required pin
        # is a required bus, so we can't do all the sanity checks we want
        #is_entire_spice_bus = (spice_name is not None) and ('width' in pin_dict)
        #is_entry_in_spice_bus, spice_bus_name, spice_bus_i = \
        #    self.check_name_for_bus(spice_name)
        #is_entry_in_req_bus, req_bus_name, req_bus_i = \
        #    self.check_name_for_bus(template_name)

        #if template_name is not None and spice_name is not None:
        #    spice_map_name = spice_bus_name if is_entry_in_spice_bus else spice_name
        #    template_map_name = req_bus_name if is_entry_in_req_bus else template_name
        #    self.spice.map(spice_map_name, template_map_name,
        #                   spice_index = spice_bus_i,
        #                   template_index = req_bus_i)

        if spice_name is not None and template_name is not None:
            self.spice.map(spice_name, template_name)

        s = SignalIn(
            value,
            type_,
            get_random,
            auto_set,
            spice_name,
            template_name,
        )
        return s



    '''
    Test class
    Every property you might want is extracted by one of the tests
    '''
    class Test():
        '''
        The sequence for a test is:
            __init__, which might set parameter algebra or calculate number of datapoints
            set_signals
            for i in range(num_points):
                testbench
            run spice simulation
            for i in range(num_points):
                analysis
            post_process
            linear regression
            post_process2

        '''
        def __init__(self):
            # we will probably edit the signals to add test-specific ones
            self.signals_in = self.signals_in.copy()

        '''
        The first three methods will almost always be overridden
        '''

        parameter_algebra = []

        def set_signals(self):
            print('Running default set_signals')
            return []

        def testbench(self):
            # usually returns a list of read objects
            return []

        def analysis(self, reads) -> dict:
            # takes testbench outputs as "reads" input
            return {}

        '''
        The remaining methods will usually not be overridden
        '''

        def get_signal_by_template_name(self, name):
            for s in self.signals_in:
                if s.template_name == name:
                    return s
            for s in self.signals_out:
                if s.template_name == name:
                    return s
            return None

        def __getattr__(self, item):
            s = self.get_signal_by_template_name(item)
            if s:
                return s
            else:
                # TODO do I return this?
                return super()


class TestbenchCreator:

    def __init__(self, template):
        self.template = template
        #self.test = test

    def get_random(self, signals):
        analog = []
        ba = []
        for s in signals:
            print(s)
            if s.get_random:
                if s.type_ == 'analog':
                    analog.append(s)
                elif s.type_ == 'binary_analog':
                    ba.append(s)
        samples = fixture.Sampler.get_orthogonal_samples(
            len(analog),
            len(ba),
            10
        )
        samples_T = list(zip(*samples))

        def scale(x, range_):
            a, b = range_
            return a + x*(b-a)

        random = {}
        for s, r in zip(analog + ba, samples_T):
            if s.type_ == 'analog':
                r = [scale(x, s.value) for x in r]
            random[s] = r
        return random


    def run(self, test):
        assert isinstance(test, self.template.Test)

        #my_signals = self.get_signals()
        # TODO don't overwrite signals
        test.set_signals()

        # get random inputs
        self.random = self.get_random(test.signals_in)
        print('got random')

class AmpTemplate(EmptyTemplate):

    class DCTest(EmptyTemplate.Test):
        parameter_algebra = {
            'amp_output': {'dcgain': 'in_single', 'offset': '1'}
        }

        def get_signals(self):
            in_value = Signal('conceptual_in_value', value=(0, 5))
            return [in_value]

    info = 42

    tests = [DCTest]


def break_bus_name(bus_name):
    '''
    'myname' -> (False, None, None)
    'myname<5> -> (True, 'myname', 5>
    'myname<5><10> -> (True, 'myname<5>', 10)
    '''
    #if name == None:
    #    return (False, None, None)

    def regex_escape(c):
        if c in '[]\\^()|':
            return '\\' + c
        else:
            return c

    # NOTE these delims are searched for with regex, so [] need escaping
    delims = ['<:>', '[:]']
    for delim in delims:
        open, mid, close = regex_escape(delim[0]), regex_escape(delim[1]), regex_escape(delim[2])
        match = re.match(f'^(.*?){open}([0-9])+:([0-9]+){close}(.*)$', bus_name)
        if match:
            bus_name = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            post = match.group(4)
            direction = 1 if end >= start else -1
            bus_name_sub = bus_name + delim[0] + '%d' + delim[2] + post
            indices = range(start, end+direction, direction)
            return (bus_name_sub, indices)
    # TODO should I wrap the bus_name in a list?
    return bus_name

def parse_cfg(circuit_config_filename):
    with open(circuit_config_filename) as f:
        d = yaml.safe_load(f)
        #d['filename'] = circuit_config_filename

        assert 'template' in d
        assert 'spice_name' in d
        assert 'spice_io' in d


        #d = {
        #    'spice_io': {
        #        'bus1<2:0>[0:1]': {'dtype': 1, 'template_name': 'req'},
        #        'sampler_clk_only': {'template_name': 'clk[0]'},
        #        'pb<0>': {'template_name': 'pb_a'}
        #    }
        #}

        # edit spice pins to break buses into their individual components
        def break_bus(pin_name, pin_dict):
            '''
            break_bus('mybus<1:0>', {abc: def, 'template_name': 'req[3]')
            ->
            {'mybus<1>: {abc: def, 'template_name': 'req[3][1]'},
             'mybus<0>: {abc: def, 'template_name': 'req[3][0]'}}
            '''
            name = break_bus_name(pin_name)
            # TODO: allow template_name to be a bus range
            if type(name) is str:
                return {pin_name: pin_dict}
            else:
                bus_name_sub, indices = name
                ans = {}
                for i in indices:
                    pin_name = bus_name_sub % i
                    pin_dict_copy = pin_dict.copy()
                    if 'template_name' in pin_dict_copy:
                        pin_dict_copy['template_name'] += f'[{i}]'
                    ans.update(break_bus(pin_name, pin_dict_copy))

                return ans

        spice_io = d['spice_io']
        new_spice_io = {}
        for pin_name in spice_io.keys():
            pin_d = spice_io[pin_name]
            if 'value' in pin_d:
                value = ast.literal_eval(str(pin_d['value']))
                pin_d['value'] = value
            # TODO extract bus delim? update: I forgot why I wanted this...
            new_spice_io.update(break_bus(pin_name, pin_d))
        print(new_spice_io)
        d['spice_io'] = new_spice_io

        return d




if __name__ == '__main__':


    parsed = parse_cfg('./tests/configs/new_parameterized_amp.yaml')

    # generate IO
    io = []
    pins = parsed['spice_io']
    for name, p in pins.items():
        dt = getattr(real_types, p['datatype'])
        #value = ast.literal_eval(str(p.get('value', None)))
        value = p.get('value', None)
        dt = dt(value)
        direction = getattr(real_types, p['direction'])
        dt = direction(dt)
        if 'width' in p:
            dt = real_types.Array(p['width'], dt)
        io += [name, dt]


    class UserCircuit(magma.Circuit):
        name = parsed['spice_name']
        IO = io

    t = fixture.SimpleAmpTemplate(parsed, UserCircuit)
    test1 = t.tests[0]()
    tc = TestbenchCreator(t)
    tc.run(test1)


    exit()

    parse_cfg('fixture/example_config.yaml')

    template_name = AmpTemplate

    tt = AmpTemplate('pin_info 456')
    c = tt.DCTest()
    c.run()
