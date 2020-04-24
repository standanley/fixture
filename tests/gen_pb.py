N = 4

def sel_pin(i):
    return f'thm_sel_bld[{i}]'

sel_pins_str = ' '.join([sel_pin(i) for i in range(N)])
subckt = f'.subckt phase_blender ph_in[1] ph_in[0] {sel_pins_str} ph_out vdd'

def inverter(i):
    n = f'Minvn{i} clk_{i}_inv {sel_pin(i)} 0 0 EENMOS w=.1u l=.1u'
    p = f'Minvp{i} clk_{i}_inv {sel_pin(i)} vdd vdd EEPMOS w=.1u l=.1u'
    return '\n'.join([f'* inv for {sel_pin(i)}', n, p])

def get_wn(i):
    return 0.1*2**i

def get_wp(i):
    return 2*get_wn(i)

def mux(i, side):
    # if sel_pins are all 0, connect to ph_in[0]
    # when connecting side 0, clk_n is inverse
    clk_n = sel_pin(i) if side else f'clk_{i}_inv'
    clk_p = sel_pin(i) if not side else f'clk_{i}_inv'
    n = f'Mmuxn{i}_{side} ph_in[{side}] {clk_n} ph_out 0 EENMOS w={get_wn(i)} l=.1u'
    p = f'Mmuxp{i}_{side} ph_in[{side}] {clk_p} ph_out vdd EENMOS w={get_wp(i)} l=.1u'
    return '\n'.join([f'* mux for ph_in[{side}], {sel_pin(i)}', n, p])

invs = ['* Inverters for clocks']
muxs = ['* Muxes']
for i in range(N):
    invs.append(inverter(i))
    invs.append('')
    muxs.append(mux(i, 0))
    muxs.append(mux(i, 1))
    muxs.append('')

inv = '\n'.join(invs)
mux = '\n'.join(muxs)

models = '''
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
'''

end = '.ends'

total = '\n'.join([subckt, inv, mux, models, end])

print(total)

    

    
