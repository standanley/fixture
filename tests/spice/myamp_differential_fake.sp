* NMOS/PMOS models from
* https://people.rit.edu/lffeee/SPICE_Examples.pdf

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt myamp inp inn outp outn vdd vss

R0 vdd outn 5000
MN0 outn inp vss vss EENMOS w=0.4u l=0.1u

R1 vdd outp 5000
MN1 outp inn vss vss EENMOS w=0.4u l=0.1u

.ends
