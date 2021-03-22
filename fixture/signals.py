import re

class SignalIn():
    def __init__(self,
                 value,
                 type_,
                 get_random,
                 auto_set,
                 spice_name,
                 spice_pin,
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
        self.spice_pin = spice_pin
        self.template_name = template_name
        #self.bus_name = bus_name
        #self.bus_i = bus_i

class SignalOut():
    def __init__(self,
                 #value,
                 type_,
                 #get_random,
                 #auto_set,
                 spice_name,
                 spice_pin,
                 template_name#,
                 #bus_name,
                 #bus_i
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

def create_signal(pin_dict):
    type_ = pin_dict.get('datatype', 'analog')
    spice_pin = pin_dict.get('spice_pin', None)
    spice_name = None if spice_pin is None else str(spice_pin)
    template_name = pin_dict.get('template_pin', None)

    if template_name is None:
        optional_types = ['real', 'binary_analog', 'true_digital']
        assert type_ in optional_types, f'Optional datatype for {spice_name} must be {optional_types}, not {type_}'

    if pin_dict['direction'] == 'input':
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


class SignalManager:
    def __init__(self, signals=None):
        if signals is None:
            signals = []

        self.signals = []
        for s in signals:
            self.add_signal(s)

    def search(self, attr, name):
        # look through signals for buses containing name
        # return a signal or python list of signals, or list of lists...
        # attr should be either spice_name or template_name
        braces_open = '[<{'
        braces_close = ']>}'
        #re_braces_open = '(' + '|'.join(re.escape(b) for b in braces_open) + ')'
        #re_braces_close= '(' + '|'.join(re.escape(b) for b in braces_close) + ')'
        re_braces_open = '[' + re.escape(braces_open) +']'
        re_braces_close= '[' + re.escape(braces_close) + ']'
        re_index = re_braces_open + '[0-9]+' + re_braces_close

        result = None

        def updated(result_old, s, indices):
            if indices == []:
                assert result_old is None
                return s
            else:
                assert result_old is None or type(result_old) == list
                if result_old is None:
                    result_old = []
                result_new = result_old + [None] * max(0, indices[0] - len(result_old) + 1)
                result_new[indices[0]] = updated(result_new[indices[0]], s, indices[1:])
                return result_new

        def parse_name(sig_name, goal_name):
            if sig_name is None:
                return None
            test = f'^{re.escape(goal_name)}(({re_index})*)$'
            m = re.match(test, sig_name)
            if not m:
                return None
            indices_str = m.group(1)
            indices_split = [g[1:-1] for g in re.findall(re_index, indices_str)]
            indices = [int(i) for i in indices_split]
            return indices

        for s in self.signals:
            indices = parse_name(getattr(s, attr), name)
            if indices is not None:
                result = updated(result, s, indices)

        return result

    def from_spice_pin(self, spice_pin):
        for s in self.signals:
            if s.spice_pin == spice_pin:
                return s
        assert False, f'No signal with spice pin {spice_pin}'

    def from_spice_name(self, spice_name):
        ans = self.search('spice_name', spice_name)
        if ans is None:
            raise KeyError(f'No signal with spice name {spice_name}')
        return ans

    def from_template_name(self, template_name):
        ans = self.search('template_name', template_name)
        if ans is None:
            raise KeyError(f'No signal with template name {template_name}')
        return ans

    def add_signal(self, s):
        # add signal to self.signals
        # used to update bus info but now that is done every query instead
        return self.signals.append(s)

    def copy(self):
        return SignalManager(self.signals)

    def __add__(self, o):
        assert isinstance(o, SignalManager)
        return SignalManager(self.signals + o.signals)

    def __iter__(self):
        return iter(self.signals)

    def __getattr__(self, item):
        try:
            return self.from_template_name(item)
        except KeyError:
            raise AttributeError

