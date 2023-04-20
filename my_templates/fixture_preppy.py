import re
import preppy

class VerilogTemplate:
    test_pattern = '\\{\\{\\s*BEGINTESTS\\s*\\}\\}(.*)\\{\\{\\s*ENDTESTS\\s*\\}\\}'
    header_pattern = '\\{\\{\\s*BEGINHEADER\\s*\\}\\}(.*)\\{\\{\\s*ENDHEADER\\s*\\}\\}'

    class ModuleParam:
        def __init__(self, text):
            pattern = '\\s*parameter (\\w+)\\s*=\\s*([\\w\\.\\-]+),?'
            parsed = re.match(pattern, text)
            if not parsed:
                assert False, f'Could not parse parameter "{text}"'

            self.name = parsed.group(1)
            self.value = parsed.group(2)

        def compile(self, param_dict):
            value = param_dict.get(self.name, self.value)
            return f'{self.name} = {self.value}'

    class ModuleIO:
        def __init__(self, text):
            # I think we can use lazy evaluation to get multi-word types?
            pattern = '^\\s*(\\w+?)\\s+(.+)\\s+(\\w+?)\\s*,?$'
            parsed = re.match(pattern, text)
            if not parsed:
                assert False, f'Could not parse io "{text}"'

            self.direction = parsed.group(1)
            self.type_ = parsed.group(2)
            self.name = parsed.group(3)

        def compile(self, template_to_circuit):
            assert self.name in template_to_circuit, f'No mapping for template io "{self.name}"'
            return f'{self.direction} {self.type_} {template_to_circuit[self.name]}'


    class FixtureParam:
        def __init__(self, text):
            pattern = '\\s*(\\w+?)\\s+(\\w+)\\s*;?'
            parsed = re.match(pattern, text)
            if not parsed:
                assert False, f'Could not parse io "{text}"'

            self.type_ = parsed.group(1)
            self.name = parsed.group(2)

        def compile(self, param_dict, mode='dummy'):
            # TODO digital modes
            # TODO types
            assert self.name in param_dict, f'No info for fixture parameter "{self.name}"'
            return param_dict[self.name]['raw_verilog']

    def __init__(self, text):
        print('todo')
        self.text = text

        # first, find test list
        tests = re.search(self.test_pattern, text, re.DOTALL)
        assert tests, 'ERROR: test pattern not found'
        self.test_list = tests.group(1).strip().split('\n')
        self.def_call = '{{def(' + ', '.join(self.test_list) + ')}}\n'

        # next, preprocess header
        header = re.search(self.header_pattern, text, re.DOTALL)
        assert header, 'ERROR: test pattern not found'
        self.header_text = header.group(1)
        test_info_temp = [True for _ in self.test_list]
        self.header = self.parse_header(test_info_temp)


    def preppy_process(self, text, test_info):
        preppy_input = self.def_call + text
        preppy_module = preppy.getModule('', sourcetext=preppy_input)
        preppy_output = preppy_module.get(*test_info)
        return preppy_output

    def parse_header(self, test_info):
        p_cap = '(.*?)'
        p_a = '^\\s*module\\s*\\('
        p_b = '\\)\\s*'
        p_c = '\\s*\\('
        p_d = '\\s*\\);?\\s*'
        p_e = '$'

        parse_header_pattern = ''.join([
            p_a,
            p_cap, # module parameters
            p_b,
            p_cap, # name
            p_c,
            p_cap, # io
            p_d,
            p_cap, # fixture params
            p_e
        ])
        header = self.preppy_process(self.header_text, test_info)

        res = re.match(parse_header_pattern, header, flags=re.DOTALL)
        if not res:
            start = 0
            for pattern in [p_a, p_b, p_c, p_d]:
                temp = re.search(pattern, header[start:], flags=re.DOTALL)
                print('searching for ', p_a, 'found', temp)
                if temp:
                    start = temp.end()
            assert False, 'Error matching header to pattern'

        module_params, name, io, fixture_params = res.groups()

        self.module_params = []
        for param_line in module_params.strip().split('\n'):
            mp = self.ModuleParam(param_line)
            self.module_params.append(mp)

        self.module_name = name

        self.module_io = []
        for io_line in io.strip().split('\n'):
            io = self.ModuleIO(io_line)
            self.module_io.append(io)

        self.fixture_params = []
        for fp_line in fixture_params.strip().split('\n'):
            if fp_line == '':
                continue
            fp = self.FixtureParam(fp_line)
            self.fixture_params.append(fp)

        print()


    def compile_header(self, dictionary):
        module_parameters = self.compile_module_parameters(dictionary)
        module_io = self.compile_module_io(dictionary)
        module_name = dictionary['module_name']
        name_mapping = self.compile_name_mapping(dictionary)
        fixture_parameters = self.compile_fixture_parameters(dictionary)

        header = '\n'.join([
            'module (',
            module_parameters,
            f') {module_name} (',
            module_io,
            ');',
            name_mapping,
            '',
            fixture_parameters
        ])
        return header

    def compile(self, dictionary):
        test_info_temp = [False for _ in self.test_list]

        # first, preprocess the header
        self.parse_header(test_info_temp)

        header_compiled = self.compile_header(dictionary)

        # next, preprocess everything else
        text_notests = re.sub(self.test_pattern, '', self.text, flags=re.DOTALL)
        text_notests_parsedheader = re.sub(self.header_pattern, header_compiled, text_notests, flags=re.DOTALL)
        model = self.preppy_process(text_notests_parsedheader, test_info_temp)
        return model


    def compile_module_parameters(self, dictionary):
        param_dict = dictionary['module_params']
        module_param_strings = [mp.compile(param_dict) for mp in self.module_params]
        return ',\n'.join(module_param_strings)

    def compile_module_io(self, dictionary):
        io_mapping = dictionary['io_mapping']
        io_strings = [io.compile(io_mapping) for io in self.module_io]
        return ',\n'.join(io_strings)

    def compile_fixture_parameters(self, dictionary):
        param_dict = dictionary['fixture_params']
        fixture_param_strings = [fp.compile(param_dict) for fp in self.fixture_params]
        return ',\n'.join(fixture_param_strings)

    def compile_name_mapping(self, dictionary):
        statements = []
        io_mapping = dictionary['io_mapping']

        for io in self.module_io:
            assert io.name in io_mapping, f'No mapping for io {io.name}'
            template, circuit = io.name, io_mapping[io.name]
            if circuit == template:
                continue

            if io.direction == 'input':
                statements.append(f'assign {template} = {circuit};')
            elif io.direction == 'output':
                statements.append(f'assign {circuit} = {template};')
            else:
                assert False, f'Do not know how to assign direction {io.direction}'

        return '\n'.join(statements)

if __name__ == '__main__':
    with open('amp2.sv') as f:
        text = f.read()

    vt = VerilogTemplate(text)
    ans = vt.compile({
        'module_params': {'nodefault': 6.0},
        'module_name': 'my_user_amp',
        'io_mapping': {'in': 'circuit_in', 'out': 'out'},
        'fixture_params': {
            'gain': {'raw_verilog': 'gain=42;'},
            'offset': {'raw_verilog': 'offset=0.6;'}}
    })

    print(ans)



