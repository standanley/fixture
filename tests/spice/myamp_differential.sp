* from ltspice
* tested with input cm=1.2v, vdd=3v

*XX1 N001 N003 N005 N002 N004 myamp
*C1 N004 0 5p
*C2 N002 0 5p
*V2 N005 0 3
*V3 N001 N006 SINE(1.2 .03 1k)
*V4 N003 N006 SINE(1.2 -.03 1k)

* block symbol definitions
.subckt myamp inn inp vdd outn outp
M1 N001 N002 0 0 EENMOS l=.1u w=20u
M2 outn inp N001 0 EENMOS l=.1u w=10u
M3 outp inn N001 0 EENMOS l=.1u w=10u
R1 vdd outn 1k
R2 vdd outp 1k
V1 N002 0 .6


* added to increase cm gain
* apparently the current source transistor M3 has no sensitivity to source-drain voltage
*R3 N001 outp 50k
*R4 N001 outn 50k
R5 N001 0 1k


* simulate FO1 cap load
Mload2 0 outn 0 0 EENMOS l=.1u w=2000u
Mload3 0 outp 0 0 EENMOS l=.1u w=2000u
C2 outp 0 1p
C3 outn 0 1p

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
.ends myamp
