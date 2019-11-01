import fixture
temp = fixture.template_master.TemplateMaster
parse = temp.parse_parameter_algebra

def test_amp():
    test = 'out_single ~ gain:in_single + offset'
    parse(test)

def test_differential():
    fs  = ['I(Outp - Outn) ~ gain:I(Inp-Inn) + cm_gain:I((inp+inn)/2) + offset']
    fs  = ['I(Outp - Outn) ~ cm_gain:I((inp+inn)/2) + offset + gain:I(Inp-Inn)']
    fs += ['I(Outp - Outn)   ~   gain  :  I(Inp-Inn)  +cm_gain:I((inp+inn)/2)+ offset']

    for f in fs:
        lhs, rhs = parse(f)
        print(lhs, rhs)
        assert lhs == 'I(Outp - Outn)'
        assert rhs['I(Inp-Inn)'] == 'gain'
        assert rhs['I((inp+inn)/2)'] == 'cm_gain'
        assert rhs['1'] == 'offset'
    



if __name__ == '__main__':
    #test_amp()
    test_differential()
