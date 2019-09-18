* Automatically generated file.
.include /home/dstanley/research/facet/tests/spice/myamp.sp
X0 in_ out vdd vss myamp
.subckt inout_sw_mod sw_p sw_n ctl_p ctl_n
    Gs sw_p sw_n cur='V(sw_p, sw_n)*(0.999999999*V(ctl_p, ctl_n)+1e-09)'
.ends
X1 __vss_v vss __vss_s 0 inout_sw_mod
V2 __vss_v 0 DC 0 PWL(0 0 2e-08 0)
V3 __vss_s 0 DC 1 PWL(0 1 2e-08 1)
X4 __vdd_v vdd __vdd_s 0 inout_sw_mod
V5 __vdd_v 0 DC 0 PWL(0 0 5e-09 0 5.2e-09 1.2 2e-08 1.2)
V6 __vdd_s 0 DC 1 PWL(0 1 5e-09 1 5.2e-09 1 2e-08 1)
X7 __in__v in_ __in__s 0 inout_sw_mod
V8 __in__v 0 DC 0 PWL(0 0 1e-08 0 1.02e-08 0.7 1.5000000000000002e-08 0.7 1.52e-08 0.8 2e-08 0.8)
V9 __in__s 0 DC 1 PWL(0 1 1e-08 1 1.02e-08 1 1.5000000000000002e-08 1 1.52e-08 1 2e-08 1)
.probe out
.ic
.tran 2.0000000000000002e-11 2e-08
.control
run
set filetype=ascii
write
exit
.endc
.end
