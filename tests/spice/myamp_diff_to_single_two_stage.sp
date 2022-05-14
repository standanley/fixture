* with fake tail bias voltage ot 0.6
* out is 1.48 + 12.0*in_diff + 0.18*cm_in, gain is 9.83
* bias voltage

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt myamp inn inp vdd out out_test
* we put the inputs in backwards here because the second stage inverts the output
Xstage1 inp inn vdd out_stage1 myamp_stage1
*Rstage1load out_stage1 0 500

* just common source with a resistor
Mgain out out_stage1 vdd vdd EEPMOS l=0.1u w=10u
*Rcurrent out 0 300
Mtail out vbias 0 0 EENMOS l=0.1u w=10u

Vbias vbias 0 0.85

* to give the transistor some drain dependence
Rtgain out vdd 1k
Rtgain2 out vss 1k

* debugging
Rtest out_stage1 out_test 1



.ends myamp

* this part of the amp is copied from myamp_diff_to_single.sp
* block symbol definitions
.subckt myamp_stage1 inn inp vdd out
* outn sets the gates for the top transistors; doesn't go out
M1 N001 N002 0 0 EENMOS l=.1u w=20u
M2 outn inp N001 0 EENMOS l=.1u w=10u
M3 out  inn N001 0 EENMOS l=.1u w=10u
Mtopn outn outn vdd vdd EEPMOS l=.1u w=10u
Mtopp out  outn vdd vdd EEPMOS l=.1u w=10u

* the stupid transistors have no built-in drain voltage sensitivity
* so you just get infinite gain all the time without these resistors
R1 vdd outn 1k
R2 vdd out 1k

* I should really make a current bias input with a mirror here
V1 N002 0 0.8


* added to increase cm gain
* apparently the current source transistor M3 has no sensitivity to source-drain voltage
*R3 N001 out  50k
*R4 N001 outn 50k
R5 N001 0 1k


* simulate FO1 cap load
*Mload2 0 outn 0 0 EENMOS l=.1u w=2000u
*Mload3 0 out  0 0 EENMOS l=.1u w=2000u
*C2 out  0 .5p
*C3 outn 0 .5p
*Rrload2p load2p out  300
*Rrload2n load2n outn 300
*Ccload2p load2p 0 .5p
*Ccload2n load2n 0 .5p


.ends myamp_stage1
