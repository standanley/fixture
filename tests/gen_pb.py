N = 16

def sel_pin(i):
    return f'thm_sel_bld[{i}]'

def ph_pin(s):
    return f'ph_in[{s}]'

#sel_pins_str = ' '.join([sel_pin(i) for i in range(N)])
#subckt = f'.subckt phase_blender {ph_pin(1)} {ph_pin(0)} {sel_pins_str} ph_out vdd'

# copy subckt from spf file so they match
subckt = '.SUBCKT phase_blender ph_in[1] ph_in[0] thm_sel_bld[15] thm_sel_bld[10] thm_sel_bld[5] thm_sel_bld[14] thm_sel_bld[13] thm_sel_bld[12] thm_sel_bld[11] thm_sel_bld[9] thm_sel_bld[8] thm_sel_bld[6] thm_sel_bld[3] thm_sel_bld[2] thm_sel_bld[1] thm_sel_bld[0] thm_sel_bld[4] thm_sel_bld[7] ph_out vdd'

def buff(i, s):
    na = f'MbufnA{i}_{s} buf_inv{i}_{s} {ph_pin(s)} 0 0 EENMOS w=.1u l=1u'
    pa = f'MbufpA{i}_{s} buf_inv{i}_{s} {ph_pin(s)} vdd vdd EEPMOS w=.2u l=1u'
    nb = f'MbufnB{i}_{s} buf{i}_{s} buf_inv{i}_{s} 0 0 EENMOS w=.01u l=1u'
    pb = f'MbufpB{i}_{s} buf{i}_{s} buf_inv{i}_{s} vdd vdd EEPMOS w=.02u l=1u'
    c = f'Cbuf{i}_{s} bif{i}_{s} 0 100n'
    return '\n'.join([f'* buffer for {ph_pin(s)}', na, pa, nb, pb])
    
    

def inverter(i):
    n = f'Minvn{i} clk_{i}_inv {sel_pin(i)} 0 0 EENMOS w=.1u l=.1u'
    p = f'Minvp{i} clk_{i}_inv {sel_pin(i)} vdd vdd EEPMOS w=.1u l=.1u'
    return '\n'.join([f'* inv for {sel_pin(i)}', n, p])

def get_wn(i):
    return 0.1#0.1*2**i

def get_wp(i):
    return 2*get_wn(i)

def mux(i, side):
    # if sel_pins are all 0, connect to ph_in[0]
    # when connecting side 0, clk_n is inverse
    clk_n = sel_pin(i) if side else f'clk_{i}_inv'
    clk_p = sel_pin(i) if not side else f'clk_{i}_inv'
    n = f'Mmuxn{i}_{side} buf{i}_{side} {clk_n} ph_out 0 EENMOS w={get_wn(i)} l=.1u'
    p = f'Mmuxp{i}_{side} buf{i}_{side} {clk_p} ph_out vdd EEPMOS w={get_wp(i)} l=.1u'
    return '\n'.join([f'* mux for ph_in[{side}], {sel_pin(i)}', n, p])

bufs = ['* Buffers for input phases']
invs = ['* Inverters for clocks']
muxs = ['* Muxes']
for i in range(N):
    bufs.append(buff(i, 0))
    bufs.append(buff(i, 1))
    bufs.append('')
    invs.append(inverter(i))
    invs.append('')
    muxs.append(mux(i, 0))
    muxs.append(mux(i, 1))
    muxs.append('')

buf= '\n'.join(bufs)
inv = '\n'.join(invs)
mux = '\n'.join(muxs)

models = '''
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
'''

end = '.ends'

total = '\n'.join([subckt, buf, inv, mux, models, end])

print(total)

    

    
