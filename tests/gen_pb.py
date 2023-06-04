N = 16
BINARY_WEIGHTING = False

# NOTE the skywater PDK seems to use um, although this file treats them as the spice default, m
#NMOS = 'sky130_fd_pr__nfet_01v8'
#PMOS = 'sky130_fd_pr__pfet_01v8_hvt'
#NMOS_UNIT = .52
#NMOS_LENGTH = 0.26
#CBUF_SIZE = '100f'
NMOS = 'EENMOS'
PMOS = 'EEPMOS'
NMOS_UNIT = 0.25
NMOS_LENGTH = 1
#PH1_MULTIPLIER = 1.5
PMOS_UNIT = NMOS_UNIT * 2
PMOS_LENGTH = NMOS_LENGTH
CBUF_SIZE = '1f'

# name for input select pin
def sel_pin(i, smooth=True):
    #s = '_smooth' if smooth else ''
    s = ''
    return f'thm_sel_bld{s}[{i}]'

# name for input phase pin
def ph_pin(s):
    return f'ph_in[{s}]'

sel_pins_str = ' '.join([sel_pin(i, False) for i in range(N)])
subckt = f'.subckt phase_blender {ph_pin(1)} {ph_pin(0)} {sel_pins_str} ph_out vdd'

# copy subckt from spf file so they match
#subckt = '.SUBCKT phase_blender ph_in[1] ph_in[0] thm_sel_bld[15] thm_sel_bld[10] thm_sel_bld[5] thm_sel_bld[14] thm_sel_bld[13] thm_sel_bld[12] thm_sel_bld[11] thm_sel_bld[9] thm_sel_bld[8] thm_sel_bld[6] thm_sel_bld[3] thm_sel_bld[2] thm_sel_bld[1] thm_sel_bld[0] thm_sel_bld[4] thm_sel_bld[7] ph_out vdd'

# double inverter from phase input to buf{i}_{s}
def buff(i, s):
    #L = 'X'
    L = 'M'
    na = f'{L}bufnA{i}_{s} buf_inv{i}_{s} {ph_pin(s)} 0 0 {NMOS} w={NMOS_UNIT}u l={NMOS_LENGTH}u'
    pa = f'{L}bufpA{i}_{s} buf_inv{i}_{s} {ph_pin(s)} vdd vdd {PMOS} w={PMOS_UNIT}u l={PMOS_LENGTH}u'
    nb = f'{L}bufnB{i}_{s} buf{i}_{s} buf_inv{i}_{s} 0 0 {NMOS} w={NMOS_UNIT*10}u l={NMOS_LENGTH}u'
    pb = f'{L}bufpB{i}_{s} buf{i}_{s} buf_inv{i}_{s} vdd vdd {PMOS} w={PMOS_UNIT*10}u l={PMOS_LENGTH}u'
    c = f'Cbuf{i}_{s} buf{i}_{s} 0 {CBUF_SIZE}'
    c2 = f'Cbuf_inv{i}_{s} buf_inv{i}_{s} 0 {CBUF_SIZE}'
    return '\n'.join([f'* buffer for {ph_pin(s)}', na, pa, nb, pb, c, c2])
    
    
# created inverted versions of select pins at clk_{i}_inv
# TODO clk is a bad name here I think
def inverter(i):
    #L = 'X'
    L = 'M'
    n = f'{L}invn{i} clk_{i}_inv {sel_pin(i)} 0 0 {NMOS} w={NMOS_UNIT}u l={NMOS_LENGTH}u'
    p = f'{L}invp{i} clk_{i}_inv {sel_pin(i)} vdd vdd {PMOS} w={PMOS_UNIT}u l={PMOS_LENGTH}u'
    return '\n'.join([f'* inv for {sel_pin(i)}', n, p])

# get width for nmos/pmos things. This is where to choose thermometer vs. binary, etc.
def get_wn(i):
    return NMOS_UNIT#0.1*2**i
def get_wp(i):
    return PMOS_UNIT

# half a mux, conditionally connects one buffered input phase to ph_out
# Acts like a switch, probably NOT the same as a standard cell mux
def mux(i, side):
    # if sel_pins are all 0, connect to ph_in[0]
    # when connecting side 0, clk_n is inverse
    clk_n = sel_pin(i) if side else f'clk_{i}_inv'
    clk_p = sel_pin(i) if not side else f'clk_{i}_inv'
    #L = 'X'
    L = 'M'
    n = f'{L}muxn{i}_{side} buf{i}_{side} {clk_n} ph_out 0 {NMOS} w={get_wn(i)}u l={NMOS_LENGTH}u'
    p = f'{L}muxp{i}_{side} buf{i}_{side} {clk_p} ph_out vdd {PMOS} w={get_wp(i)}u l={PMOS_LENGTH}u'
    return '\n'.join([f'* mux for ph_in[{side}], {sel_pin(i)}', n, p])

bufs = ['* Buffers for input phases']
invs = ['* Inverters for clocks']
muxs = ['* Muxes']
fix = ['* Maybe this will fix convergence issues']
for i in range(N):
    bufs.append(buff(i, 0))
    bufs.append(buff(i, 1))
    bufs.append('')
    invs.append(inverter(i))
    invs.append('')
    muxs.append(mux(i, 0))
    muxs.append(mux(i, 1))
    muxs.append('')
    fix.append(f'Rfix0_{i} thm_sel_bld[{i}] thm_sel_bld_smooth[{i}] 1')
    fix.append(f'Cfix1_{i} thm_sel_bld_smooth[{i}] 0 1f')
    fix.append('')

buf= '\n'.join(bufs)
inv = '\n'.join(invs)
mux = '\n'.join(muxs)
#fix = '\n'.join(fix)
fix = ''

models = '''
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
'''
#models = '.lib "../sky130/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice" tt'

end = '.ends'

debug = []
#debug.append('* DEBUG: snap output to buf0_0')
#debug.append('Edebug ph_out 0 buf0_0 0 1')
debug = '\n'.join(debug)

total = '\n\n'.join([models, subckt, debug, fix, buf, inv, mux, end])

print(total)

    

    
