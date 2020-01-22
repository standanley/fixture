* NMOS/PMOS models from
* https://people.rit.edu/lffeee/SPICE_Examples.pdf

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt vco adj out vdd vss

* n0 -> n1 -> out -> n0
* NAME drain gate source bulk

MN0 n1 n0_delay vss vss EENMOS w=100.0u l=0.1u
MP0 n1 n0_delay adj vdd EEPMOS w=200.0u l=0.1u

MN1 out n1_delay vss vss EENMOS w=100.0u l=0.1u
MP1 out n1_delay adj vdd EEPMOS w=200.0u l=0.1u

MN2 n0 out_delay vss vss EENMOS w=100.0u l=0.1u
MP2 n0 out_delay adj vdd EEPMOS w=200.0u l=0.1u

R0 n0 n0_delay 1e3
C0 n0_delay vss 1e-9

R1 n1 n1_delay 1e3
C1 n1_delay vss 1e-9

Rout out out_delay 1e3
Cout out_delay vss 1e-9

.ends
