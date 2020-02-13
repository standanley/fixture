* NMOS/PMOS models from
* https://people.rit.edu/lffeee/SPICE_Examples.pdf

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt myamp adj inp inn outp outn vdd vss

* internal vdd gets shut off depending on en
*MPswitch vdd_int en vdd vdd EEPMOS w=4u l=0.1u
V0 vdd_int vdd 0

R0 vdd_int outn_raw 5000
MN0 outn_raw inp vss vss EENMOS w=0.4u l=0.1u

R1 vdd_int outp_raw 5000
MN1 outp_raw inn vss vss EENMOS w=0.4u l=0.1u

* bump up common mode output by 0.42*adj
Eadjp outp_0 outp_raw adj vss 1
Eadjn outn_0 outn_raw adj vss 1

Ebuffp vss outp vss outp_0 1
Ebuffn vss outn vss outn_0 1
.ends
