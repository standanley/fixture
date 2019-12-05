.subckt myphaseblender gnd input_a input_b output sel<2> sel<1> sel<0> vdd

M1 N001 sel<0> input_a input_a EENMOS w=1u l=0.1u
M2 N001 sel<1> input_a input_a EENMOS w=1u l=0.1u
M3 N001 sel<2> input_a input_a EENMOS w=3u l=0.1u
M4 N001 sel<0>_bar input_b input_b EENMOS w=1u l=0.1u
M5 N001 sel<1>_bar input_b input_b EENMOS w=1u l=0.1u
M6 N001 sel<2>_bar input_b input_b EENMOS w=3u l=0.1u
M7 N002 N001 gnd gnd EENMOS w=5u l=0.1u
M8 output N002 gnd gnd EENMOS w=5u l=0.1u
C1 N001 gnd 1n
M14 sel<0>_bar sel<0> gnd gnd EENMOS w=5u l=0.1u
M16 sel<1>_bar sel<1> gnd gnd EENMOS w=5u l=0.1u
M18 sel<2>_bar sel<2> gnd gnd EENMOS w=5u l=0.1u
M19 sel<2>_bar sel<2> vdd vdd EEPMOS w=10u l=.1u
M11 N001 sel<0>_bar input_a input_a EEPMOS w=2u l=.1u
M12 N001 sel<1>_bar input_a input_a EEPMOS w=2u l=.1u
M13 N001 sel<2>_bar input_a input_a EEPMOS w=6u l=.1u
M20 N001 sel<0> input_b input_b EEPMOS w=2u l=.1u
M21 N001 sel<1> input_b input_b EEPMOS w=2u l=.1u
M22 N001 sel<2> input_b input_b EEPMOS w=6u l=.1u
C2 sel<2>_bar gnd 1n
M15 sel<1>_bar sel<1> vdd vdd EEPMOS w=10u l=.1u
*M17 sel<0>_bar sel<0> vdd vdd EEPMOS w=10u l=.1u
M17 vdd sel<0> sel<0>_bar vdd EEPMOS w=10u l=.1u
M9 N002 N001 vdd vdd EEPMOS w=10u l=.1u
M10 output N002 vdd vdd EEPMOS w=10u l=.1u
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.ends

