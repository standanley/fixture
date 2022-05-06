* from ltspice
* tested with input cm=1.4-1.6v, vdd=3v

* with fake tail bias voltage ot 0.6 and fake load (in the subckt)
* amp_output	offset	const_1	2.002e+00
* amp_output	dcgain_invec[0]	const_1	1.705e+01
* amp_output	dcgain_invec[1]	const_1	-5.024e-02

*XX1 N001 N003 N005 N002 N004 myamp
*C1 N004 0 5p
*C2 N002 0 5p
*V2 N005 0 3
*V3 N001 N006 SINE(1.2 .03 1k)
*V4 N003 N006 SINE(1.2 -.03 1k)

* block symbol definitions
.subckt myamp inn inp vdd out
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


** simulate FO1 cap load
Mload2 0 outn 0 0 EENMOS l=.1u w=2000u
Mload3 0 out  0 0 EENMOS l=.1u w=2000u
C2 out  0 .5p
C3 outn 0 .5p
Rrload2p load2p out  300
Rrload2n load2n outn 300
Ccload2p load2p 0 .5p
Ccload2n load2n 0 .5p
*Rload1 out 0 500
*Rload2 out 0 1000


.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
.ends myamp
