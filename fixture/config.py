import fixture
import fault
import sys

header = '''
# Generated configuration file
# Please fill in anything marked <TODO>

'''
TAB = '    '

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
    if len(template.required_ports) > 0:
        lines.append('# TODO Tag pins with each of the following required inputs/outputs:')
        for name in template.required_ports:
            lines.append(f'#{TAB*2}template_pin: {name}')

    lines.append('pins:')
    for pin_name in ports.keys():
        lines.append(f'{TAB}{pin_name}:')
        lines.append(f'{TAB*2}direction: <TODO "input" or "output">')
        lines.append(f'{TAB*2}datatype: <TODO "real", "digital", or "binary_analog">')
        lines.append(f'{TAB*2}<TODO possibly add "value" or "template_pin" and/or "width">')

    lines.append('')
    for k, v in template.required_info.items():
        lines.append(f'{k}: <TODO {v}>')

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
    make_config(spice_filename, template_name)

