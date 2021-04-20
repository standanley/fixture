import fixture
import fault
import sys
import html

header = '''
# Generated configuration file
# Please fill in anything marked <TODO>

'''

html_header = '''
<head> <meta charset="UTF-8"> </head>
<h1> This is the fixture interactive config generator</h1>
<h2> Just fill out the form and then copy the config file at the bottom </h2>

'''
TAB = '    '

def make_config_interactive(spice_filename, template_name, skip_writing_file=False):
    form = []
    config = []
    js = []
    js_startup = []

    def form_entry(id):
        return f'<span id="{id}"></span>'

    def radio(id, title, options):
        ''' takes care of form and js'''
        form_id = id + '_radio'
        form.append('\n')
        form.append(f'<label for="{form_id}">{title}</label>')
        for option in options:
            form.append(f'<input type="radio" name="{form_id}" value={option} onchange="{id}_fun()"> {option} </input>')
        form.append('\n<br>')

        js.append('\n')
        js.append(f'function {id}_fun() {{')
        js.append(f'{TAB}buttons = document.getElementsByName("{form_id}");')
        js.append(f'{TAB}value="TODO";')
        js.append(f'{TAB}buttons.forEach( button => {{')
        js.append(f'{TAB}{TAB}if(button.checked) {{ value = button.value; }}')
        js.append(f'{TAB}}});')
        js.append(f'{TAB}document.getElementById("{id}").innerHTML = value;')
        js.append(f'}}')

        js_startup.append(f'{id}_fun();')

    def dropdown(id, title, options):
        ''' takes care of form and js '''
        form_id = id + '_select'
        form_name = id + '_select_name'
        form.append('')
        form.append(f'<label for="{form_id}">{title}</label>')
        form.append(f'<select id="{form_id}" name="{form_name}" onchange="{id}_fun()">')
        for option in options:
            form.append(f'  <option value="{option}"> {option} </option>')
        form.append(f'</select> <br>')

        js.append('')
        js.append(f'function {id}_fun() {{')
        js.append(f'{TAB}value = document.getElementById("{form_id}").value;')
        js.append(f'{TAB}document.getElementById("{id}").innerHTML= value;')
        js.append(f'}}')

        js_startup.append(f'{id}_fun();')

    def text(id, title, optional_key=None):
        ''' takes care of form and js '''
        form_id = id + '_text'
        form.append('')
        form.append(f'<label for="{form_id}">{title}</label>')
        form.append(f'<input type="text" id="{form_id}" oninput="{id}_fun()"/> <br>')

        js.append('')
        js.append(f'function {id}_fun() {{')
        js.append(f'{TAB}value = document.getElementById("{form_id}").value;')
        js.append(f'{TAB}form_span = document.getElementById("{id}");')
        if optional_key is None:
            js.append(f'{TAB}form_span.innerHTML = value;')
        else:
            js.append(f'{TAB}if (value=="") {{form_span.innerHTML = value;}}')
            # NOTE this always inserts an extra blank line ... which is okay for my usage so far
            js.append(f'{TAB}else {{form_span.innerHTML = "        {optional_key}: "+value+"\\n";}}')
        js.append(f'}}')

        js_startup.append(f'{id}_fun();')



    template = getattr(fixture.templates, template_name)
    circuit = fault.spice_target.DeclareFromSpice(spice_filename)

    circuit_name = circuit.name
    ports = circuit.IO.ports

    # circuit name
    config.append(f'circuit_name: {circuit_name}')

    # pins
    config.append('\n# pins')
    for pin_name in ports.keys():

        pin_name_clean = html.escape(pin_name).replace('&', '_').replace(';', '_')

        form.append('<br>')
        config.append(f'{TAB}{pin_name}:')

        direction_id = pin_name_clean + '_direction'
        config.append(f'{TAB}{TAB}direction: {form_entry(direction_id)}')
        radio(direction_id, f'"{pin_name}" Direction:', ['input', 'output'])

        type_id = pin_name_clean + '_type'
        config.append(f'{TAB}{TAB}datatype: {form_entry(type_id)}')
        type_options = ['TODO', 'bit', 'real', 'analog', 'binary_analog', 'true_digital']
        dropdown(type_id, f'"{pin_name}" Data Type:', type_options)

        value_id = pin_name_clean + '_value'
        config.append(form_entry(value_id))
        text(value_id, f'"{pin_name}" Value (float for pinned value, pair of floats for range, or blank): ', 'value')

    # mapping
    pin_choices = ['TODO', *ports.keys()]
    form.append('<br>')
    config.append('')
    config.append('# Mapping between template and user pins')
    config.append('template_pins:')
    for required in template.required_ports:
        required_clean = html.escape(required).replace('&', '_').replace(';', '_')
        map_name = required_clean + '_map'

        config.append(f'{TAB}{required}: {form_entry(map_name)}')
        dropdown(map_name, f'Circuit port that corresponds to required port "{required}": ', pin_choices)


    # extras
    form.append('<br>')
    config.append('')
    config.append('# Additional template-required information')
    config.append('extras:')
    for k, v in template.required_info.items():
        extra_id = k + '_extra'
        config.append(f'{TAB}{html.escape(k)}: {form_entry(extra_id)}')
        text(extra_id, v)

    temp = '''
function test() {
        var newWindow = window.open("name.yaml", "_blank");
        debugger;
        newWindow.document.write("test:\\n    attr: yep\\n");
    }

    test();
'''

    form_text = '<form>\n    ' + '\n    '.join(form) + '\n</form>'
    config_text = '<hr><br>\n<pre>\n' + '\n'.join(config) + '\n</pre>'
    js_text = '<script type="text/javascript">\n    ' + '\n    '.join(js+js_startup)+temp + '\n</script>'

    everything = '\n\n'.join([html_header, form_text, config_text, js_text])

    if not skip_writing_file:
        with open(f'{circuit_name}_config_helper.html', 'w') as f:
            f.write(everything)

    return everything


def make_config(spice_filename, template_name, skip_writing_file=False):
    template = getattr(fixture.templates, template_name)
    circuit = fault.spice_target.DeclareFromSpice(spice_filename)

    circuit_name = circuit.name
    ports = circuit.IO.ports
    #for name in ports.keys():
    #    print(name)

    lines = [header]
    lines.append(f'name: {circuit_name}')
    lines.append(f'filepath: {spice_filename}')
    lines.append(f'template: {template_name}')
    lines.append('')

    # pins
    lines.append('pins:')
    for pin_name in ports.keys():
        lines.append(f'{TAB}{pin_name}:')
        lines.append(f'{TAB*2}direction: <TODO "input" or "output">')
        lines.append(f'{TAB*2}datatype: <TODO "real", "digital", or "binary_analog">')

    # mapping
    if len(template.required_ports) > 0:
        lines.append('# TODO Tag each of the following required inputs/outputs with a pin')
        lines.append('template_pins:')
        for name in template.required_ports:
            lines.append(f'#{TAB}{name}: <TODO>')

    # extras
    lines.append('')
    lines.append('extras:')
    for k, v in template.required_info.items():
        lines.append(f'{TAB}{k}: <TODO {v}>')

    lines.append('')
    lines.append('test_config_file: <TODO>')

    text = '\n'.join(lines)

    if not skip_writing_file:
        with open(f'{circuit_name}_config.yaml', 'w') as f:
            f.write(text)
    return text


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print('Must specify spice_filename and template_name as command line arguments')
    spice_filename = args[1]
    template_name = args[2]
    text = make_config_interactive(spice_filename, template_name)


