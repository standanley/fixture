# sampler with N outputs
# each switch has an nmos and a pmos
# clkn controls nmos, clkp controls pmos

N = 4

CGS_N = '10f'
CGD_N = '10f'
CDS_N = '10f'
CGS_P = '10f'
CGD_P = '10f'
CDS_P = '10f'
COUT = '50000f'
WN = '10u'
WP = '20u'
R_SRC = '50'
DEBUG = 'in_nonideal'

header = '''
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
'''
print(header)


clks = ' '.join([f'clkn<{i}>' for i in range(N)] + [f'clkp<{i}>' for i in range(N)])
outs = ' '.join(f'out<{i}>' for i in range(N))
subckt = f'.subckt sampler7 {clks} in_ {outs} vdd vssi debug'

print(subckt)

print('\n* Mock 50 Ohm input driver')
print(f'Rdriver in_ in_nonideal {R_SRC}')


print('\n* sampling NMOS')
for i in range(N):
    print(f'Msampn{i} out<{i}> clkn<{i}> in_nonideal vss EENMOS w={WN} l=0.1u')
    print(f'Cgsn{i} clkn<{i}> out<{i}> {CGS_N}')
    print(f'Cgdn{i} clkn<{i}> in_nonideal {CGD_N}')
    print(f'Cdsn{i} in_nonideal out<{i}> {CDS_N}')

print('\n* sampling PMOS')
for i in range(N):
    print(f'Msampp{i} out<{i}> clkp<{i}> in_nonideal vdd EEPMOS w={WP} l=0.1u')
    print(f'Cgsp{i} clkp<{i}> out<{i}> {CGS_P}')
    print(f'Cgdp{i} clkp<{i}> in_nonideal {CGD_P}')
    print(f'Cdsp{i} in_nonideal out<{i}> {CDS_P}')

print('\n* output cap')
for i in range(N):
    print(f'Cout{i} out<{i}> 0 {COUT}')

print('\n* debug')
print(f'Rdebug debug {DEBUG} 0')

print()
print('.ends sampler7')

