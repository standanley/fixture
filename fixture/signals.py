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

class SignalOut():
    def __init__(self,
                 #value,
                 type_,
                 #get_random,
                 #auto_set,
                 spice_name,
                 template_name#,
                 #bus_name,
                 #bus_i
                 ):
        # TODO: do we need bus info?
        self.type_ = type_
        # TODO auto read?
        #self.auto_set = auto_set
        self.spice_name = spice_name
        self.template_name = template_name
        #self.bus_name = bus_name
        #self.bus_i = bus_i

def create_signal(pin_dict):
    type_ = pin_dict.get('type', 'analog')
    spice_name = pin_dict.get('spice_name', None)
    template_name = pin_dict.get('template_pin', None)

    if pin_dict['direction'] == 'input':
        value = pin_dict.get('value', None)
        get_random = pin_dict.get('get_random',
                                  (type(value) == tuple) or (type_ == 'binary_analog' and value == None))
        auto_set = pin_dict.get('auto_set',
                                get_random or type(value) == int or type(value) == float)

        s = SignalIn(
            value,
            type_,
            get_random,
            auto_set,
            spice_name,
            template_name,
        )
        return s

    elif pin_dict['direction'] == 'output':
        s = SignalOut(type_,
                      spice_name,
                      template_name)
        return s
    else:
        assert False, 'Unrecognized pin direction' + pin_dict['direction']