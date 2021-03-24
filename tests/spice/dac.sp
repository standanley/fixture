* NMOS/PMOS models from
* https://people.rit.edu/lffeee/SPICE_Examples.pdf

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt dac input<0> input<1> outputp outputn vdd vss

M0 outputp input<0> vdd vdd EEPMOS w=1u l=0.1u
M1 outputp input<1> vdd vdd EEPMOS w=2u l=0.1u

R1 outputp vss 5k

R2 outputn vss 1k

.ends dac
