import re
import numbers
import numpy as np
from functools import reduce

class SignalIn():
    def __init__(self,
                 value,
                 nominal,
                 type_,
                 get_random,
                 auto_set,
                 spice_name,
                 spice_pin,
                 template_name,
                 optional_expr,
                 representation=None
            ):
        # TODO: do we need bus info?
        self.value = value
        self.nominal = nominal
        self.type_ = type_
        self.get_random = get_random
        self.auto_set = auto_set
        self.spice_name = spice_name
        self.spice_pin = spice_pin
        self.template_name = template_name
        self.optional_expr = optional_expr
        self.representation = representation

    def __str__(self):
        return f'<{str(self.template_name)} / {self.spice_name}>'

    def friendly_name(self):
        return self.template_name if self.spice_name is None else self.spice_name


    def __getstate__(self):
        d = self.__dict__.copy()
        if d['spice_pin'] is not None:
            d['spice_pin'] = d['spice_name']
        return d

    def __setstate__(self, state):
        # TODO don't have access to dut?
        self.__dict__ = state


class CenteredSignalIn:
    def __init__(self, ref):
        self.ref = ref

        def name(n):
            return None if n is None else f'{n}_centered'
        self.spice_name = name(ref.spice_name)
        self.template_name = name(ref.template_name)

    def __str__(self):
        return f'<{str(self.template_name)} / {self.spice_name}>'


class SignalOut():
    def __init__(self,
                 type_,
                 spice_name,
                 spice_pin,
                 template_name,
                 auto_measure,
                 representation=None
                 ):
        self.type_ = type_
        self.spice_name = spice_name
        self.spice_pin = spice_pin
        self.template_name = template_name
        self.auto_measure = auto_measure
        self.representation = representation

    def __str__(self):
        return f'<{str(self.template_name)} / {self.spice_name}>'

    def friendly_name(self):
        return self.template_name if self.spice_name is None else self.spice_name

    def __getstate__(self):
        d = self.__dict__.copy()
        if d['spice_pin'] is not None:
            d['spice_pin'] = d['spice_name']
        return d

    def __setstate__(self, state):
        # TODO don't have access to dut?
        self.__dict__ = state


def create_signal(pin_dict, c_name=None, c_pin=None, t_name=None):
    type_ = pin_dict.get('datatype', 'analog')
    assert (c_name is None) == (c_pin is None)
    spice_pin = c_pin
    spice_name = c_name
    template_name = t_name

    if pin_dict['direction'] == 'input':
        is_proxy_component = pin_dict.get('is_proxy_component', False)
        if template_name is None:
            pinned = isinstance(pin_dict.get('value', None), numbers.Number)
            optional_types = ['analog', 'binary_analog', 'true_digital']
            assert (pinned
                    or is_proxy_component
                    or type_ in optional_types), f'Optional datatype for {spice_name} must be {optional_types}, not {type_}'

        value = pin_dict.get('value', None)
        get_random = pin_dict.get('get_random',
                                  not is_proxy_component
                                  and template_name is None and (
                                  (type(value) == tuple) or (type_ == 'binary_analog' and value == None)))
        def guess_nominal(value, type_):
            if value is None:
                return None
            elif isinstance(value, numbers.Number):
                return value
            elif len(value) == 1:
                return value[0]
            elif len(value) == 2 and type_ == 'analog':
                assert len(value) == 2
                return sum(value) / 2
            elif len(value) == 2 and type_ == 'binary_analog':
                assert len(value) == 2
                #return int(sum(value) // 2)
                # this is likely a single bit of a bus, and the nominal will be
                # set when the bus is put together
                return None
            else:
                assert ValueError(f'Cannot guess nominal value for <{c_name}/{t_name}>')

        nominal = pin_dict.get('nominal', guess_nominal(value, type_))
        auto_set = pin_dict.get('auto_set',
                                get_random or type(value) == int or type(value) == float)
        optional_expr = get_random

        s = SignalIn(
            value,
            nominal,
            type_,
            get_random,
            auto_set,
            spice_name,
            spice_pin,
            template_name,
            optional_expr
        )
        return s

    elif pin_dict['direction'] == 'output':
        s = SignalOut(type_,
                      spice_name,
                      spice_pin,
                      template_name,
                      (template_name is None
                       and not pin_dict.get('is_proxy_component', False)))
        return s
    else:
        assert False, 'Unrecognized pin direction' + pin_dict['direction']

def create_input_domain_signal(name, value, spice_pin=None,
                               optional_expr=False):
    return SignalIn(
        value,
        None,
        'analog',
        type(value) == tuple,
        spice_pin is not None,
        spice_pin,
        None if spice_pin is None else str(spice_pin),
        name,
        optional_expr
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
            indices_parsed.append((x,))
            info_parsed.append(index[0] + index[-1] + 'a')
        else:
            m = re.match(re_index_range_groups, index)
            s, e = int(m.group(2)), int(m.group(3))
            indices_parsed.append((s, e))
            direction = 'a' if e >= s else 'd'
            info_parsed.append(m.group(1) + m.group(4) + direction)

    def names_flat(indices_details):
        if len(indices_details) == 0:
            return [(bus_name, ())]
        elif len(indices_details[-1][0]) == 1:
            (num,), index_info = indices_details[-1]
            base = names_flat(indices_details[:-1])
            postfix = (index_info[0] + str(num) + index_info[1], num)
            ans = []
            for basename, baseindices in base:
                ans.append((basename + postfix[0], baseindices + (postfix[1],)))
            return ans
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

        self.signals_by_circuit_name = {}
        for s_or_a in self.signals:
            if isinstance(s_or_a, SignalArray):
                for s in s_or_a.flatten():
                    if s.spice_name is not None:
                        self.signals_by_circuit_name[s.spice_pin] = s

                if s_or_a.spice_name is not None:
                    self.signals_by_circuit_name[s_or_a.spice_name] = s_or_a
            else:
                if s_or_a.spice_name is not None:
                    self.signals_by_circuit_name[s_or_a.spice_name] = s_or_a

        if signals_by_template_name is None:
            self.signals_by_template_name = {}
        else:
            self.signals_by_template_name = signals_by_template_name

    def add(self, signal):
        # TODO should we allow SignalArray here?
        assert isinstance(signal, (SignalIn, SignalOut))
        #assert signal.spice_name is None
        #assert signal.template_name is not None

        self.signals.append(signal)
        if signal.template_name is not None:
            self.signals_by_template_name[signal.template_name] = signal
        if signal.spice_name is not None:
            self.signals_by_circuit_name[signal.spice_name] = signal


    def copy(self):
        # the intention is for the template writer to add signals to the copy
        # without changing the original
        signals_copy = self.signals.copy()
        by_template_copy = self.signals_by_template_name.copy()
        # by_circuit_name will be rebuilt automatically
        return SignalManager(signals_copy, by_template_copy)

    def from_template_name(self, name):
        # return a Signal or SignalArray of signals according to template name
        bus_name, indices = parse_name(name)
        if len(indices) == 0:
            return self.signals_by_template_name[name]
        else:
            a = self.signals_by_template_name[bus_name]
            s_or_ss = a[tuple(indices)]
            return s_or_ss

    def from_circuit_pin(self, pin):
        # pin is a magma object
        for x in self.signals:
            if isinstance(x, SignalArray):
                for s in x.flatten():
                    if s.spice_pin is pin:
                        return s
            else:
                if x.spice_pin is pin:
                    return x

    def from_circuit_name(self, name):
        # return a Signal or SignalArray of signals according to template name
        bus_name, indices = parse_name(name)
        if len(indices) == 0:
            return self.signals_by_circuit_name[name]
        else:
            a = self.signals_by_circuit_name[bus_name]
            s_or_ss = a[tuple(indices)]
            return s_or_ss

    def from_str(self, name):
        # TODO this is based on assumptions about the various __str__ methods
        if name[0] != '<' or name[-1] != '>':
            raise KeyError(name)
        name = name[1:-1]
        tokens = name.split(' / ')
        if len(tokens) == 2:
            t_name, c_name = tokens
        else:
            t_name, c_name, indices = tokens

        if t_name != 'None':
            return self.from_template_name(t_name)
        elif c_name != 'None':
            return self.from_circuit_name(c_name)
        else:
            assert False, 'looking for signal without name'


    def inputs(self):
        for s in self.signals:
            if isinstance(s, SignalIn):
                yield s
            elif isinstance(s, SignalArray):
                all_in = all(isinstance(x, SignalIn) for x in s.flatten())
                all_out = all(isinstance(x, SignalOut) for x in s.flatten())
                if all_in:
                    yield s
                else:
                    assert all_out, f'Mixed input and output in {s}'

    def random(self):
        # return a list of Signals (always analog) and SignalArrays (always qa)
        # basically we want to organize these the way they will be randomly
        # sampled; so each object in the list is one dimension
        ans = set()
        for x in self.signals:
            if isinstance(x, SignalIn):
                if x.get_random:
                    ans.add(x)
            elif isinstance(x, SignalArray):
                if x.type_ == 'binary_analog':
                    assert x.get_random == True, 'qa that is not random?'
                    ans.add(x)
                else:
                    # we can't include it as one SA, but we should check bits
                    for bit in x.flatten():
                        assert bit.type_ != 'binary analog', f'mixed qa/not in {x}'
                        if getattr(bit, 'get_random', None):
                            ans.add(bit)
        return list(sorted(ans, key=lambda s: s.friendly_name()))


    def random_qa(self):
        # TODO right now it returns the whole SignalArray if it's full of qa,
        # but will return individual bits if they're not in a SA or mixed in
        def check_s(s):
            return (isinstance(s, SignalIn)
                    and s.get_random
                    and s.type_ in ['binary_analog', 'bit'])

        for s_or_a in self.signals:
            if isinstance(s_or_a, SignalArray):
                if all(check_s(s) for s in s_or_a.flatten()):
                    yield s_or_a
                else:
                    for s in s_or_a.flatten():
                        if check_s(s):
                            yield s
            elif isinstance(s_or_a, SignalIn):
                if check_s(s_or_a):
                    yield s_or_a

    def true_digital(self):
        # return a list of signals - no SignalArrays
        ans = [s for s in self.flat() if s.type_ == 'true_digital']
        return ans

    def optional_expr(self):
        ans = [x for x in self.signals if getattr(x, 'optional_expr', None)]
        for x in ans:
            assert x.type_ in ['analog', 'binary_analog']
        return ans

    def optional_quantized_analog(self):
        for x in self.optional_expr():
            assert x.type_ in ['analog', 'binary_analog']
        for x in self.signals:
            if isinstance(x, SignalArray):
                assert x.type_ is not None
        return [x for x in self.optional_expr() if x.type_ == 'binary_analog']

    def optional_true_analog(self):
        for x in self.optional_expr():
            assert x.type_ in ['analog', 'binary_analog']
        ans = [x for x in self.optional_expr() if x.type_ == 'analog']
        return ans

    def auto_measure(self):
        for s in self.flat():
            if isinstance(s, SignalOut) and s.auto_measure:
                yield s

    def vectored_out(self):
        ans = []
        for s in self.signals:
            if (isinstance(s, SignalArray)
                    and s.representation is not None
                    and s.representation.style == 'vector'
                    and isinstance(s[0], SignalOut)):
                ans.append(s)
        return ans

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
        try:
            return self.from_template_name(item)
        except KeyError:
            raise AttributeError


class SignalArray:
    attributes_from_children = ['type_',
                                'get_random',
                                'auto_set',
                                'spice_name',
                                'spice_pin',
                                'template_name',
                                'optional_expr',
                                ]

    def __init__(self, signal_array, info, template_name=None, spice_name=None,
                 is_vectored_input=False):
        self.array = signal_array
        self.info = info
        self.template_name = template_name
        self.spice_name = spice_name
        self.is_vectored_input = is_vectored_input
        self.representation = None

        # TODO nested buses
        def guess_value():
            low = self.get_decimal_value([0]*len(self.array))
            high = self.get_decimal_value([1]*len(self.array))
            return (low, high)
        self.value = self.info.get('value', guess_value())
        self.nominal = self.info.get('nominal')
        if self.nominal is None and self.value is not None:
            # we can guess the nominal
            if isinstance(self.value, int):
                self.nominal = self.value
            elif len(self.value) == 1:
                self.nominal = self.value[0]
            elif len(self.value) == 2:
                self.nominal = sum(self.value) // 2
            else:
                raise ValueError(f'Could not guess nominal for <{self.template_name}/{self.spice_name}>')
        if self.nominal is not None:
            # distribute to children
            binary = self.get_binary_value(self.nominal)
            for b, s in zip(binary, self.array):
                s.nominal = b


    def map(self, fun):
        return np.vectorize(fun)(self.array)

    def get_decimal_value(self, data):
        assert self.info['datatype'] == 'binary_analog'
        if isinstance(data, dict):
            data = [data[s] for s in self.array]
            data = np.array(data)
        else:
            data = np.array(data)

        if self.info['bus_type'] == 'binary' or self.info['bus_type'] == 'binary_exact':
            # TODO I don't like making the assumption here, but it should only affect plots, not functionality
            first_one = self.info.get('first_one', 'low')
            assert first_one in ['low', 'high']
            def to_dec(seq):
                seq_ordered = seq if first_one == 'low' else seq[::-1]
                val = sum(2**i*x for i, x in enumerate(seq_ordered))
                return val
        elif self.info['bus_type'] == 'signed_magnitude':
            # NOTE we assume the sign bit is the highest-order, i.e. not the "first_one"
            first_one = self.info.get('first_one', 'low')
            assert first_one in ['low', 'high']
            def to_dec(seq):
                if first_one == 'low':
                    seq = seq[::-1]
                sign, seq = seq[0], seq[1:]
                s = ''.join(str(x) for x in seq)
                magnitude = int(s, 2)
                sign = 1 if sign == 0 else -1
                return sign * magnitude
        else:
            assert False, 'TODO'

        if len(data.shape) == 1:
            return to_dec(data)
        else:
            to_dec_vec = np.vectorize(to_dec, signature='(n)->()')
            ans = to_dec_vec(data.T)
            return ans

    def get_binary_value(self, decimal):
        assert self.info['datatype'] in ['binary_analog', 'bit']
        # need to allow int and np.int64
        assert isinstance(decimal, numbers.Integral)

        if self.info['bus_type'] == 'binary':
            first_one = self.info.get('first_one', None)
            assert first_one in ['low', 'high'], f'{self} must set first_one to be "low" or "high"'
            ans_str = bin(decimal)[2:]
            ans = [0]*(self.shape[0] - len(ans_str)) + [int(x) for x in ans_str]
            if first_one == 'low':
                ans = ans[::-1]
            return ans
        elif self.info['bus_type'] == 'signed_magnitude':
            assert False, 'TODO'
        else:
            assert False, 'TODO'


    def __getitem__(self, key):
        slice = self.array[key]
        if isinstance(slice, np.ndarray):
            # TODO should slice inherit the info? For the info we currently use, no
            return SignalArray(slice, {})
        else:
            return slice

    def __getattr__(self, name):
        # if you're stuck in a loop here, self probably has no .array
        if name in self.attributes_from_children:
            # If every signal in this array agrees, return that. Otherwise None
            entries = map(lambda s: getattr(s, name, None), self.array.flatten())
            ans = reduce(lambda a, b: a if a == b else None, entries)
            return ans
        else:
            return getattr(self.array, name)

    def __setattr__(self, name, value):
        # TODO
        if False and name in self.attributes_from_children:
            for child in self.array.flatten():
                child.__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    # TODO I don't think these are used any more
    def __getstate__(self):
        d = self.__dict__.copy()
        ndarray = d['array']

        def listify(x):
            if isinstance(x, np.ndarray):
                return [listify(y) for y in x]
            else:
                return x
        array = listify(ndarray)

        d['array'] = array
        return d

    def __setstate__(self, state):
        self.array = None
        state['array'] = np.array(state['array'], dtype=object)
        self.__dict__ = state

    def __str__(self):
        return f'<{str(self.template_name)} / {self.spice_name} / {self.array.shape}>'

    def friendly_name(self):
        return self.template_name if self.spice_name is None else self.spice_name
