/****************************************************************

Copyright (c) 2018 Stanford University. All rights reserved.

The information and source code contained herein is the 
property of Stanford University, and may not be disclosed or
reproduced in whole or in part without explicit written 
authorization from Stanford University.

* Filename   : amplifier.template.sv
* Author     : Byongchan Lim (bclim@stanford.edu)
* Description: SV template for an amplifier cell

* Note       :

* Todo       :
  - 

* Revision   :
  - 00/00/00 : 

****************************************************************/
/*******************************************************
* An amplifier with possible output equalization
* - Input referred voltage offset as a static parameter
* - Gain Compression
* - Dynamic behavior (a pole or two-poles with a zero)
* - 

* Calibrating metrics:
* 1. Av = gm*Rout 
* 2. Max output swing = Itail*Rout 
* 3. fp1, fp2, fz1
*******************************************************/

$$#$${
$$#import os
$$#from dave.common.misc import *
$$#def_file = os.path.join(os.environ['DAVE_INST_DIR'], 'dave/mgenero/api_mgenero.py')
$$#api_fullpath = get_abspath(def_file, True, None)
$$#api_dir = get_dirname(api_fullpath)
$$#api_base = os.path.splitext(get_basename(api_fullpath))[0]
$$#
$$#import sys
$$#if not api_fullpath in sys.path:
$$#  import sys
$$#  sys.path.append(api_dir)
$$#from api_mgenero import *
$$#
$$#}$$

`include "mLingua_pwl.vh"

module $$(Module.name()) #(
  $$(Module.parameters())
) (
  $$(Module.pins())
);


`protect
//pragma protect 
//pragma protect begin

`get_timeunit
PWLMethod pm=new;

// about to map
$$Pin.print_map() $$# map between user pin names and generic ones
// just mapped

//----- BODY STARTS HERE -----

//----- SIGNAL DECLARATION -----
pwl ONE = `PWL1;
pwl ZERO = `PWL0;

// pwl v_id_lim;   // limited v_id 
// pwl v_oc; // output common-mode voltage
// pwl v_od; // output differential voltage
// pwl vid_max, vid_min; // max/min of v_id for slewing 
// pwl vop, von;
// pwl v_od_filtered;
// pwl vop_lim, von_lim;
// pwl v_id, v_icm; // differential and common-mode inputs


// TODO is t0 ever used? does mLingua need it?
real t0;
// real v_icm_r;
// real vdd_r;
$$PWL.declare_optional_analog_pins_in_real()

// real fz1, fp1, fp2; // at most, two poles and a zero
// real Av;    // voltage gain (gm*Rout)
// real max_swing; // Max voltage swing of an output (Itail*Rout)
// real vid_r; // vid<|vid_r| (max_swing/Av)
// real v_oc_r;  // common-mode output voltage

// Declaration of params
real gain;
real offset;

event wakeup;

//----- FUNCTIONAL DESCRIPTION -----

initial ->> wakeup; // dummy event for ignition at t=0

//-- System's parameter calculation

// discretization of control inputs
$$#PWL.instantiate_pwl2real_optional_analog_pins(['vss'] if Pin.is_exist('vss') else [])

// updating parameters as control inputs/mode inputs change

$${
# sensitivity list of always @ statement
# sensitivity = ['v_icm_r', 'vdd_r', 'wakeup'] + get_sensitivity_list() 
sensitivity = ['wakeup', 'sel'] + get_sensitivity_list() 

# model parameter mapping for back-annotation
# { testname : { test output : Verilog variable being mapped to } }
#model_param_map = { 'test1': {'gain': 'Av', 'offset_to_cm': 'v_oc_r'} }
#test = {'gain_sel[%d]'%i:'gain%d'%i for i in range(4)}
test = {'gain': 'gain'}
test['offset'] = 'offset'
model_param_map = { 'test1': test }

iv_map = {'test123':'test123'}
}$$

always @($$print_sensitivity_list(sensitivity)) begin
  t0 = `get_time;

// TODO do I need to annotate modelparam?
$$annotate_modelparam(model_param_map, iv_map)

end

//-- Model behaviors



event go_high;
event go_low;
real period;
real rising_a;
real diff;
real delay_periods;
real delay;

always @(posedge(in_a)) begin
    period = `get_time - rising_a;
    rising_a = `get_time;
end

always @(posedge(in_b)) begin
    diff = `get_time - rising_a;
    delay_periods = diff / period * gain + offset + 1;
    delay = delay_periods * period;
    #(delay) ->> go_high;
    #(period/2) ->> go_low;
end

always @(go_high) out = 1;
always @(go_low) out = 0;

//pragma protect end
`endprotect


endmodule
