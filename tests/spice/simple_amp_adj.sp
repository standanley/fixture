* NMOS/PMOS models from
* https://people.rit.edu/lffeee/SPICE_Examples.pdf

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt myamp in_ out vdd vss vbias cm_adj<3> cm_adj<2> cm_adj<1> cm_adj<0>

* mirror
*Mbias vbias vbias vdd vdd EEPMOS w=0.1u l=0.1u
Mtop  out   vbias vdd vdd EEPMOS w=1u l=0.1u
* transistor model seems to have no rout
R0 vdd out 5k

* amp
MN0 out in_ vss vss EENMOS w=1u l=0.1u
R1 out vss 20k

* adj
Madj0 out cm_adj<0> vss vss EENMOS w=0.001u l=0.1u
Madj1 out cm_adj<1> vss vss EENMOS w=0.002u l=0.1u
Madj2 out cm_adj<2> vss vss EENMOS w=0.004u l=0.1u
Madj3 out cm_adj<3> vss vss EENMOS w=0.008u l=0.1u

*Rdummy0 cm_adj<0> dummy 1k
*Rdummy1 cm_adj<1> dummy 1k
*Rdummy2 cm_adj<2> dummy 1k
*Rdummy3 cm_adj<3> dummy 1k

.ends
