.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

* block symbol definitions
.subckt myamp_stage inn_internal inp_internal vdd vbias outn outp
M1 vtail vbias 0 0 EENMOS l=.1u w=20u
M2 outn inp_internal vtail 0 EENMOS l=.1u w=10u
M3 outp inn_internal vtail 0 EENMOS l=.1u w=10u
R1 vdd outn 5k
R2 vdd outp 5k


* added to increase cm gain
* apparently the current source transistor M3 has no sensitivity to source-drain voltage
*R3 N001 outp 50k
*R4 N001 outn 50k
R5 vtail 0 10k


** simulate FO1 cap load
*Mload2 0 outn 0 0 EENMOS l=.1u w=2000u
*Mload3 0 outp 0 0 EENMOS l=.1u w=2000u
*C2 outp 0 .5p
*C3 outn 0 .5p
*Rrload2p load2p outp 300
*Rrload2n load2n outn 300
*Ccload2p load2p 0 .5p
*Ccload2n load2n 0 .5p


*Rtest1 outn outn_test 1
*Rtest2 outp outp_test 1


.ends myamp_stage


.subckt myamp inn inp vdd ibias outn outp
* just need to generate vbias, let's multiply by 10, so nominal ibias maybe 20u? no idea
Mbias ibias ibias 0 0 EENMOS l=0.1u w=2u
X1 inn inp vdd ibias outn_mid outp_mid myamp_stage
X2 outn_mid outp_mid vdd ibias outn outp myamp_stage

Rrfn inn outn 10k
Rrfp inp outp 10k
Ccfn inn outn 100p
Ccfp inp outp 100p
.ends myamp
