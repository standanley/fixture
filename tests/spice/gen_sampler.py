N = 2

header = '''
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
'''
print(header)

ignore = ['clk_v2tb', 'clk_v2t_e', 'clk_v2t_eb', 'clk_v2t_l', 'clk_v2t_lb',
    'clk_v2t_gated', 'clk_v2tb_gated']

ignore = [ f'{s}<{i}>' for s in ignore for i in range(N)]
ignore.sort()

#print('\n'.join(ignore))

clks = ' '.join(f'clk_v2t<{i}>' for i in range(N))
outs = ' '.join(f'out<{i}>' for i in range(N))
ignores = ' '.join(ignore)
subckt = f'.subckt sampler4 {clks} {ignores} in_ {outs} vdd z_debug'

print(subckt)

print('\n* sampling NMOS')
for i in range(N):
    print(f'Msamp{i} discharge_{i} clk_v2t<{i}> in_ 0 EENMOS w=1u l=0.1u')

print('\n* discharge RC')
for i in range(N):
    print(f'Rdischarge{i} discharge_{i} 0 10000000')
    print(f'Cdischarge{i} discharge_{i} 0 10p')

print('\n* trigger inverter')
for i in range(N):
    print(f'Minvn{i} 0   discharge_{i} trigger_{i} 0   EENMOS w=1u   l=0.1u')
    print(f'Minvp{i} vdd discharge_{i} trigger_{i} vdd EEPMOS w=0.5u l=0.1u')

print('\n* output NOR')
for i in range(N):
    print(f'Mnorn1{i} 0 trigger_{i}  out<{i}> 0 EENMOS w=1u l=0.1u')
    print(f'Mnorn2{i} 0 clk_v2t<{i}> out<{i}> 0 EENMOS w=1u l=0.1u')
    print(f'Mnorp1{i} vdd trigger_{i} nor_temp_{i} vdd    EEPMOS w=1u l=0.1u')
    print(f'Mnorp2{i} nor_temp_{i}    clk_v2t<{i}> out<{i}> vdd EEPMOS w=1u l=0.1u')

print('\n* output cap')
for i in range(N):
    print(f'Cout{i} out<{i}> 0 1f')

print('\n* ignore caps')
for i, s in enumerate(ignore + ['vdd', 'z_debug']):
    print(f'Cignore{i} {s} 0 1f')

print('\n* add some effect for late clock')
print('Cslow slow 0 5p')
print('Mslow slow clk_v2t_l discharge 0 EENMOS w=1u l=0.1u')

print('\n* debug')
print('Rdebug discharge_0 z_debug 0')

print()
print('.ends sampler4')

