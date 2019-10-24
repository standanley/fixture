* NMOS/PMOS models from
* https://people.rit.edu/lffeee/SPICE_Examples.pdf

.subckt my_comparator gnd my_in vdd my_out
R1 vdd N001 5k
M1 N001 my_in 0 0 EENMOS w=1u l=0.1u
R2 vdd my_out 5k
M2 my_out N001 0 0 EENMOS w=1u l=0.1u
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)
.ends my_comparator
