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

    class FixtureParam:
        def __init__(self, text):
            pattern = '\\s*(\\w+?)\\s+(\\w+)\\s*;?'
            parsed = re.match(pattern, text)
            if not parsed:
                assert False, f'Could not parse io "{text}"'

            self.type_ = parsed.group(1)
            self.name = parsed.group(2)

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
        module_parameters =  self.compile_module_parameters(dictionary)
        module_io =  self.compile_module_io(dictionary)
        module_name = dictionary['module_name']
        fixture_parameters =  self.compile_fixture_parameters(dictionary)
        header = '\n'.join([
            'module (',
            module_parameters,
            f') {module_name} ('
            module_io,
            ');'
            fixture_parameters
        ])

    def compile(self, dictionary):
        test_info_temp = [True for _ in self.test_list]

        # first, preprocess the header
        self.parse_header(test_info_temp)

        header_tag = 'HEADER_PLACEHOLDER_TAG'

        # next, preprocess everything else
        text_notests = re.sub(self.test_pattern, '', self.text, flags=re.DOTALL)
        text_notests_noheader = re.sub(self.header_pattern, header_tag, text_notests, flags=re.DOTALL)
        model_noheader = self.preppy_process(text_notests_noheader, test_info_temp)

        print(preppy_output2)



if __name__ == '__main__':
    with open('amp2.sv') as f:
        text = f.read()

    VerilogTemplate(text)



