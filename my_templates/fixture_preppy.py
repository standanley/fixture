import re
import preppy

class VerilogTemplate:
    def __init__(self, text):
        print('todo')

if __name__ == '__main__':
    with open('amp2.sv') as f:
        text = f.read()

    print(text)

    # first, find test list
    test_pattern = '\\{\\{\\s+BEGINTESTS\\s+\\}\\}(.*)\\{\\{\\s+ENDTESTS\\s+\\}\\}'
    #test_pattern = '\\{\\{hi\\}\\}'

    tests = re.search(test_pattern, text, re.DOTALL)
    assert tests, 'ERROR: test pattern not found'
    test_list = tests.group(1).strip().split('\n')
    print(test_list)
    def_call = '{{def(' + ', '.join(test_list) + ')}}\n'
    test_info = [True for _ in test_list]

    # next, preprocess header

    header_pattern = '\\{\\{\\s+BEGINHEADER\\s+\\}\\}(.*)\\{\\{\\s+ENDHEADER\\s+\\}\\}'
    header = re.search(header_pattern, text, re.DOTALL)
    assert header, 'ERROR: test pattern not found'
    header_text = header.group(1)
    preppy_input1 = def_call + header_text
    preppy_module1 = preppy.getModule('', sourcetext=preppy_input1)
    preppy_output1 = preppy_module1.get(*test_info)

    print(preppy_output1)


    print('into preppy 2\n')
    text_notests = re.sub(test_pattern, '', text, flags=re.DOTALL)
    text_notests_noheader = re.sub(header_pattern, '', text_notests, flags=re.DOTALL)
    print(text_notests_noheader)
    preppy_input2 = def_call + text_notests_noheader
    preppy_module2 = preppy.getModule('', sourcetext=preppy_input2)
    preppy_output2 = preppy_module2.get(*test_info)

    print(preppy_output2)

