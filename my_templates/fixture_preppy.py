import re
import preppy

class VerilogTemplate:
    start_test_pattern = '\\{\\{\\s*BEGINTESTS\\s*\\}\\}'
    end_test_pattern = '\\{\\{\\s*ENDTESTS\\s*\\}\\}'
    start_header_pattern = '\\{\\{\\s*BEGINHEADER\\s*\\}\\}'
    end_header_pattern = '\\{\\{\\s*ENDHEADER\\s*\\}\\}'
    header_start_tag = 'HEADER_START_TAG'
    header_end_tag = 'HEADER_END_TAG'

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
            return f'{self.name} = {value}'

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


    @classmethod
    def parse_tests(cls, text):
        test_pattern = cls.start_test_pattern + '(.*)' + cls.end_test_pattern
        tests = re.search(test_pattern, text, re.DOTALL)
        assert tests, 'ERROR: test pattern not found'
        test_list = tests.group(1).strip().split('\n')
        test_list = [test.strip() for test in test_list]
        test_list = [test for test in test_list if test != '']
        return test_list


    def __init__(self, text, dictionary):
        # init accepts the whole dictionary, but it only looks at
        # dictionary['included_tests'], so it can be used to determine required
        # things for this template
        self.original_text = text
        self.test_info = dictionary['included_tests']

        # first, find test list
        self.test_list = self.parse_tests(self.original_text)
        self.def_call = '{{def(' + ', '.join(self.test_list) + ')}}\n'

        # remove test list
        test_pattern = self.start_test_pattern + '.*' + self.end_test_pattern
        text_notests = re.sub(test_pattern, '', self.original_text, flags=re.DOTALL)

        # replace header tags with ones that preppy won't get confused by
        text_temp = re.sub(self.start_header_pattern, self.header_start_tag, text_notests)
        text_ready = re.sub(self.end_header_pattern, self.header_end_tag, text_temp)

        # next, preprocess with preppy
        self.text_processed = self.preppy_process(text_ready, self.test_info)

        # now process the header
        header_pattern = self.header_start_tag + '(.*)' + self.header_end_tag
        header_match = re.search(header_pattern, self.text_processed, re.DOTALL)
        if not header_match:
            assert False, 'Couldnt find header, internal error?'
        header = header_match.group(1)
        # parse_header sets some object variables
        self.parse_header(header)



        #header = re.search(self.header_pattern, text, re.DOTALL)
        #assert header, 'ERROR: test pattern not found'
        #self.header_text = header.group(1)
        #test_info_temp = [True for _ in self.test_list]
        #self.header = self.parse_header(test_info_temp)


    def preppy_process(self, text, test_info):
        preppy_input = self.def_call + text
        preppy_module = preppy.getModule('', sourcetext=preppy_input)
        args = [test_info[t] for t in self.test_list]
        preppy_output = preppy_module.get(*args)
        return preppy_output

    def parse_header(self, header):
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
        #header = self.preppy_process(self.header_text, test_info)

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
        # after init we have already run preppy, but we still need to
        # compile and replace the header
        test_info = dictionary['included_tests']
        assert test_info == self.test_info, f'Model {self} was created with test info {self.test_info}, but compiled with info {test_info}'

        header_compiled = self.compile_header(dictionary)
        header_pattern = self.header_start_tag + '(.*)' + self.header_end_tag
        model = re.sub(header_pattern, header_compiled, self.text_processed, flags=re.DOTALL)
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
        return '\n'.join(fixture_param_strings)

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

    dictionary = {
        'included_tests': {'clamping': False},
        'module_params': {'nodefault': 6.0},
        'module_name': 'my_user_amp',
        'io_mapping': {'in': 'circuit_in', 'out': 'out'},
        'fixture_params': {
            'gain': {'raw_verilog': 'gain=42;'},
            'offset': {'raw_verilog': 'offset=0.6;'}}
    }
    vt = VerilogTemplate(text, dictionary)
    ans = vt.compile(dictionary)

    print(ans)



