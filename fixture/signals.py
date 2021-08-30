import re
import numbers
import numpy as np

class SignalIn():
    def __init__(self,
                 value,
                 type_,
                 get_random,
                 auto_set,
                 spice_name,
                 spice_pin,
                 template_name,

            ):
        # TODO: do we need bus info?
        self.value = value
        self.type_ = type_
        self.get_random = get_random
        self.auto_set = auto_set
        self.spice_name = spice_name
        self.spice_pin = spice_pin
        self.template_name = template_name
        #self.bus_name = bus_name
        #self.bus_i = bus_i

    def __str__(self):
        return f'<{str(self.template_name)} / {self.spice_name}>'

class SignalOut():
    def __init__(self,
                 #value,
                 type_,
                 #get_random,
                 #auto_set,
                 spice_name,
                 spice_pin,
                 template_name
                 ):
        # TODO: do we need bus info?
        self.type_ = type_
        # TODO auto read?
        #self.auto_set = auto_set
        self.spice_name = spice_name
        self.spice_pin = spice_pin
        self.template_name = template_name
        #self.bus_name = bus_name
        #self.bus_i = bus_i

def create_signal(pin_dict, c_name=None, c_pin=None, t_name=None):
    type_ = pin_dict.get('datatype', 'analog')
    assert (c_name is None) == (c_pin is None)
    spice_pin = c_pin
    spice_name = c_name
    template_name = t_name

    if pin_dict['direction'] == 'input':
        if template_name is None:
            pinned = isinstance(pin_dict.get('value', None), numbers.Number)
            optional_types = ['analog', 'binary_analog', 'true_digital']
            assert pinned or type_ in optional_types, f'Optional datatype for {spice_name} must be {optional_types}, not {type_}'

        value = pin_dict.get('value', None)
        get_random = pin_dict.get('get_random',
                                  template_name is None and (
                                  (type(value) == tuple) or (type_ == 'binary_analog' and value == None)))
        auto_set = pin_dict.get('auto_set',
                                get_random or type(value) == int or type(value) == float)

        s = SignalIn(
            value,
            type_,
            get_random,
            auto_set,
            spice_name,
            spice_pin,
            template_name,
        )
        return s

    elif pin_dict['direction'] == 'output':
        s = SignalOut(type_,
                      spice_name,
                      spice_pin,
                      template_name)
        return s
    else:
        assert False, 'Unrecognized pin direction' + pin_dict['direction']

def create_input_domain_signal(name, value, spice_pin=None):
    return SignalIn(
        value,
        'analog',
        type(value) == tuple,
        spice_pin is not None,
        spice_pin,
        None if spice_pin is None else str(spice_pin),
        name
    )


# TODO is it good practice to have all these variables in the signal namespace?
braces_open = '[<{'
braces_close = ']>}'
# re_braces_open = '(' + '|'.join(re.escape(b) for b in braces_open) + ')'
# re_braces_close= '(' + '|'.join(re.escape(b) for b in braces_close) + ')'
re_braces_open = '[' + re.escape(braces_open) + ']'
re_braces_close = '[' + re.escape(braces_close) + ']'
re_num = '[0-9]+'
re_index = re_braces_open + re_num + re_braces_close
re_index_range = re_braces_open + re_num + ':' + re_num + re_braces_close
re_index_range_groups = f'({re_braces_open})({re_num}):({re_num})({re_braces_close})'

def parse_bus(name):
    '''
    'myname'  ->  ('myname', [], [], ['myname'])
    'myname{1:3}<1:0>[4]'  ->  ('myname', [3,1,4], ['{}a', '<>d', '[]a'], [...])
    The final list is a flat list of all the bits and their locaitons
    e.g. [('myname{1}<1>[4]', (1,1,4)), ('myname{1}<0>[4]', (1,0,4)), ...]
    '''

    indices = re.findall(re_index + '|' + re_index_range, name)
    bus_name = re.sub(re_index + '|' + re_index_range, '', name)
    indices_parsed = []
    #indices_limits_parsed = []
    info_parsed = []
    for index in indices:
        m = re.match(re_index, index)
        if m is not None:
            x = int(index[1:-1])
            indices_parsed.append(x)
            #indices_limits_parsed.append(x+1)
            info_parsed.append(index[0] + index[-1] + 'a')
        else:
            m = re.match(re_index_range_groups, index)
            s, e = int(m.group(2)), int(m.group(3))
            indices_parsed.append((s, e))
            #indices_limits_parsed.append(max(s, e)+1)
            direction = 'a' if e >= s else 'd'
            info_parsed.append(m.group(1) + m.group(4) + direction)

    def names_flat(indices_details):
        if len(indices_details) == 0:
            return [(bus_name, ())]
        else:
            (s, e), index_info = indices_details[-1]
            base = names_flat(indices_details[:-1])
            nums = range(s, e+1) if e>=s else range(s, e-1, -1)
            postfixes = [(index_info[0] + str(n) + index_info[1], n) for n in nums]
            ans = []
            for basename, baseindices in base:
                ans += [(basename + pn, baseindices + (pi,))
                        for pn, pi in postfixes]
            return ans
    names = names_flat(list(zip(indices_parsed, info_parsed)))

    return bus_name, indices_parsed, info_parsed, names

def parse_name(name):
    indices = re.findall(re_index + '|' + re_index_range, name)
    bus_name = re.sub(re_index + '|' + re_index_range, '', name)
    indices_parsed = []
    for index in indices:
        m = re.match(re_index, index)
        if m is not None:
            x = int(index[1:-1])
            indices_parsed.append(x)
        else:
            m = re.match(re_index_range_groups, index)
            s, e = int(m.group(2)), int(m.group(3))
            indices_parsed.append((s, e))
    return bus_name, tuple(indices_parsed)

def expanded(name):
    '''
    'myname'  ->  'myname'
    'myname{0:2}<1:0>[4]'  ->  [['myname{0}<1>[4]', 'myname{0}<0>[4]'], ['myname{1}<1>[4]', ...], ...]
    :param name: name which may or may not be a bus
    :return: appropriately nested list of pins in bus
    '''
    indices = re.findall(re_index + '|' + re_index_range, name)
    bus_name = re.sub(re_index + '|' + re_index_range, '', name)
    indices_parsed = []
    for index in indices:
        m = re.match(re_index, index)
        if m is not None:
            indices_parsed.append((index[0], index[1:-1], index[-1]))
        else:
            m = re.match(re_index_range_groups, index)
            indices_parsed.append((m.group(1), m.group(2), m.group(3), m.group(4)))

    def make_names(prefix, indices):
        if len(indices) == 0:
            return prefix
        index = indices[0]
        if len(index) == 3:
            # don't create a list for this one
            return make_names(f'{prefix}{index[0]}{index[1]}{index[2]}', indices[1:])
        else:
            start = int(index[1])
            end = int(index[2])
            direction = 1 if end >= start else -1
            i_s = range(start, end+direction, direction)
            return [make_names(f'{prefix}{index[0]}{i}{index[3]}', indices[1:]) for i in i_s]

    return bus_name, make_names(bus_name, indices_parsed)



class SignalManager:
    def __init__(self, signals, signals_by_template_name):
        if signals is None:
            self.signals = []
        else:
            self.signals = signals

        # TODO this is broken if there are non-circuit signals here
        self.signals_by_circuit_name = {}
        for s_or_a in self.signals:
            if isinstance(s_or_a, SignalArray):
                token_signal = s_or_a.flatten()[0]
                self.signals_by_circuit_name[token_signal.spice_name] = s_or_a
            else:
                self.signals_by_circuit_name[s_or_a.spice_name] = s_or_a

        if signals_by_template_name is None:
            self.signals_by_template_name = {}
        else:
            self.signals_by_template_name = signals_by_template_name

    def add(self, signal):
        # TODO should we allow SignalArray here?
        assert isinstance(signal, (SignalIn, SignalOut))
        assert signal.spice_name is None
        assert signal.template_name is not None

        self.signals.append(signal)
        self.signals_by_template_name[signal.template_name] = signal

    def copy(self):
        # the intention is for the template writer to add signals to the copy
        # without changing the original
        signals_copy = self.signals.copy()
        #by_circuit_copy = self.signals_by_circuit_name.copy()
        by_teplate_copy = self.signals_by_template_name.copy()
        return SignalManager(signals_copy, by_teplate_copy)

    def template(self, name):
        # return a Signal or SignalArray of signals according to template name
        bus_name, indices = parse_name(name)
        if len(indices) == 0:
            return self.signals_by_template_name[name]
        else:
            a = self.signals_by_template_name[bus_name]
            s_or_ss = a[tuple(indices)]
            return s_or_ss


        pass

    def circuit(self, name):
        # return a Signal or SignalArray of signals according to circuit name
        pass

    def random(self):
        # return a list of Signals (always analog) and SignalArrays (always qa)
        assert False, 'Do we need this?'

    def random_analog(self):
        def check_s(s):
            return s.get_random and s.type_ in ['analog', 'real']

        for s_or_a in self.signals:
            if isinstance(s_or_a, SignalArray):
                # TODO whole bus at once
                for s in s_or_a.flatten():
                    if check_s(s):
                        yield s
            elif isinstance(s_or_a, SignalIn):
                if check_s(s_or_a):
                    yield s_or_a

    def random_qa(self):
        def check_s(s):
            return s.get_random and s.type_ in ['binary_analog', 'bit']

        for s_or_a in self.signals:
            if isinstance(s_or_a, SignalArray):
                # TODO whole bus at once
                for s in s_or_a.flatten():
                    if check_s(s):
                        yield s
            elif isinstance(s_or_a, SignalIn):
                if check_s(s_or_a):
                    yield s_or_a

    def auto_set(self):
        # return a list of signals
        pass

    def true_digital(self):
        # return a list of signals
        pass

    def linear_input(self):
        # list of optional inputs, signals (a) or SignalArrays (ba)
        pass

    def binary_analog(self):
        # TODO get rid of this
        return {}
    def true_analog(self):
        # TODO get rid of this
        return []


    def flat(self):
        signals = []
        for s_or_a in self.signals:
            if isinstance(s_or_a, SignalArray):
                signals += list(s_or_a.flatten())
            else:
                signals.append(s_or_a)
        return signals

    def __iter__(self):
        return iter(self.signals)

    def __getattr__(self, item):
        #if item == 'signals_in':
        #    return (s for s in self.signals if isinstance(s, SignalIn))
        #if item == 'signals_out':
        #    return (s for s in self.signals if isinstance(s, SignalOut))
        try:
            return self.template(item)
        except KeyError:
            raise AttributeError


class SignalArray:

    def __init__(self, signal_array, info):
        self.array = signal_array
        self.info = info

    #def flat(self):
    #    ss = []
    #    for key in self.order:
    #        x = self.map[key]
    #        if isinstance(x, [SignalIn, SignalOut]):
    #            ss.append(x)
    #        else:
    #            ss += x.flat()
    #    return ss

    def map(self, fun):
        return np.vectorize(fun)(self.array)
        #return np.array(map(fun, self.array))

    def __getitem__(self, key):
        slice = self.array[key]
        if isinstance(slice, np.ndarray):
            # TODO should slice inherit the info? For the info we currently use, no
            return SignalArray(slice, {})
        else:
            return slice

    def __getattr__(self, item):
        # if you're stuck in a loop here, self probably has no .array
        return getattr(self.array, item)

    #def __getattr__(self, name):
    #    return getattr(self.token_item, name)


