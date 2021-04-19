import re
import ast

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
        match = re.match(f'^(.*?){open}([0-9]+):([0-9]+){close}(.*)$', bus_name)
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


def edit_cfg(d):

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
                if 'template_pin' in pin_dict_copy:
                    pin_dict_copy['template_pin'] += f'[{i}]'
                ans.update(break_bus(pin_name, pin_dict_copy))

            return ans

    spice_io = d['pin']
    new_spice_io = {}
    for pin_name in spice_io.keys():
        pin_d = spice_io[pin_name]
        if 'value' in pin_d:
            value = ast.literal_eval(str(pin_d['value']))
            pin_d['value'] = value
        # TODO extract bus delim? update: I forgot why I wanted this...
        new_spice_io.update(break_bus(pin_name, pin_d))
    print(new_spice_io)
    d['pin'] = new_spice_io