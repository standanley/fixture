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
        # is_entire_spice_bus = (spice_name is not None) and ('width' in pin_dict)
        # is_entry_in_spice_bus, spice_bus_name, spice_bus_i = \
        #    self.check_name_for_bus(spice_name)
        # is_entry_in_req_bus, req_bus_name, req_bus_i = \
        #    self.check_name_for_bus(template_name)

        # if template_name is not None and spice_name is not None:
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