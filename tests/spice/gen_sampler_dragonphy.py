N = 1

header = '''
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
'''
print(header)

ignore = ['clk_v2tb', 'clk_v2t_e', 'clk_v2t_eb', 'clk_v2t_l', 'clk_v2t_lb',
    'clk_v2t_gated', 'clk_v2tb_gated']

#ignore = [ f'{s}' for s in ignore for i in range(N)]
ignore += ['Vcal']
ignore += [f'ctl<{x}>' for x in range(32)]
ignore.sort()

#print('\n'.join(ignore))

#clks = ' '.join(f'clk_v2t' for i in range(N))
#outs = ' '.join(f'v2t_out' for i in range(N))
#ignores = ' '.join(ignore)

io = ignore + ['clk_v2t', 'v2t_out', 'Vin', 'vdd']
io.sort()
io_string = ' '.join(x for x in io)
subckt = f'.subckt sampler5 {io_string}'

print(subckt)

print('\n* sampling NMOS')
for i in range(N):
    print(f'Msamp discharge clk_v2t Vin 0 EENMOS w=1u l=0.1u')

print('\n* discharge RC')
for i in range(N):
    print(f'Rdischarge discharge 0 10000000')
    print(f'Cdischarge discharge 0 10p')

print('\n* trigger inverter')
for i in range(N):
    print(f'Minvn 0   discharge trigger 0   EENMOS w=1u   l=0.1u')
    print(f'Minvp vdd discharge trigger vdd EEPMOS w=0.5u l=0.1u')

print('\n* output NOR')
for i in range(N):
    print(f'Mnorn1 0 trigger  v2t_out 0 EENMOS w=1u l=0.1u')
    print(f'Mnorn2 0 clk_v2t v2t_out 0 EENMOS w=1u l=0.1u')
    print(f'Mnorp1 vdd trigger nor_temp vdd    EEPMOS w=1u l=0.1u')
    print(f'Mnorp2 nor_temp    clk_v2t v2t_out vdd EEPMOS w=1u l=0.1u')

print('\n* output cap')
for i in range(N):
    print(f'Cout v2t_out 0 1f')

print('\n* ignore caps')
for i, s in enumerate(ignore):
    print(f'Cignore_{i} {s} 0 1f')

print('\n* add some effect for late clock')
print('Cslow slow 0 5p')
print('Mslow slow clk_v2t_l discharge 0 EENMOS w=1u l=0.1u')

print('\n* debug')
print('*Rdebug discharge_0 z_debug 0')

print()
print('.ends sampler4')

