import fixture
temp = fixture.regression.Regression
parse = temp.parse_parameter_algebra

def test_amp():
    test = 'out_single ~ gain:in_single + offset'
    res = parse(test)
    print(res)

def test_differential():
    fs  = ['I(outp - outn) ~ gain:I(inp-inn) + cm_gain:I((inp+inn)/2) + offset']
    fs  = ['I(outp - outn) ~ cm_gain:I((inp+inn)/2) + offset + gain:I(inp-inn)']
    fs += ['I(outp - outn)   ~   gain  :  I(inp-inn)  +cm_gain:I((inp+inn)/2)+ offset']

    for f in fs:
        lhs, rhs = parse(f)
        print(lhs, rhs)
        assert lhs == 'I(outp - outn)'
        assert rhs['I(inp-inn)'] == 'gain'
        assert rhs['I((inp+inn)/2)'] == 'cm_gain'
        assert rhs['1'] == 'offset'
    



if __name__ == '__main__':
    test_amp()
    #test_differential()
