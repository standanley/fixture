
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
* Drain Gate Source Bulk

* FIRST a differential amp
* block symbol definitions
.subckt myamp inn inp vdd outn outp
* tested with input cm=1.2v, vdd=3v
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
C2 outp 0 .5p
C3 outn 0 .5p
Rrload2p load2p outp 300
Rrload2n load2n outn 300
Ccload2p load2p 0 .5p
Ccload2n load2n 0 .5p


.ends myamp

* inv
.subckt myinv in out vdd vss
Mp out in vdd vdd EEPMOS l=.1u w=1u
Mn out in vss vss EENMOS l=.1u w=1u
.ends


* a switch
.subckt myswitch in out ctrl vdd vss
Xinv ctrl ctrl_bar vdd vss myinv
Mp out ctrl_bar in vdd EEPMOS l=.1u w=10u
Mn out ctrl     in vss EENMOS l=.1u w=10u
.ends

* NOW the sampler
.subckt mysampler_differential inp inn outp outn clk vdd vss
Rcm1 vdd cmin 180
Rcm2 vss cmin 120
Csampp swinpo ampinn 20p
Csamn swinno ampinp 20p
Xamp ampinn ampinp vdd outn outp myamp

Xswinp inp swinpo swinctrl vdd vss myswitch
Xswinn inn wninno swinctrl vdd vss myswitch
Xswcmp cmin ampinn clk vdd vss myswitch
Xswcmn cmin ampinp clk cdd css myswitch
Xswfp outp swinpo swfctrl vdd vss myswitch
Xswfn outn swinno swfctrl vdd vss myswitch

Xinvctrlb ctrl ctrl_bar vdd vss myinv
Xinvctrlsamp ctrl_bar swinctrl vdd vss myinv
Xinvfeedback swinctrl swfctrl vdd vss myniv
.ends



