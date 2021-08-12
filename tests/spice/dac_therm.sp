* NMOS/PMOS models from
* https://people.rit.edu/lffeee/SPICE_Examples.pdf

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt dac input<0> input<1> input<2> input<3> input<4> input<5> input<6> input<7> input<8> input<9> input<10> input<11> input<12> input<13> input<14> input<15> outputp outputn vdd vss

M0 outputp input<0> vdd vdd EEPMOS w=1u l=0.1u
M1 outputp input<1> vdd vdd EEPMOS w=1u l=0.1u
M2 outputp input<2> vdd vdd EEPMOS w=1u l=0.1u
M3 outputp input<3> vdd vdd EEPMOS w=1u l=0.1u
M4 outputp input<4> vdd vdd EEPMOS w=1u l=0.1u
M5 outputp input<5> vdd vdd EEPMOS w=1u l=0.1u
M6 outputp input<6> vdd vdd EEPMOS w=1u l=0.1u
M7 outputp input<7> vdd vdd EEPMOS w=1u l=0.1u
M8 outputp input<8> vdd vdd EEPMOS w=1u l=0.1u
M9 outputp input<9> vdd vdd EEPMOS w=1u l=0.1u
M10 outputp input<10> vdd vdd EEPMOS w=1u l=0.1u
M11 outputp input<11> vdd vdd EEPMOS w=1u l=0.1u
M12 outputp input<12> vdd vdd EEPMOS w=1u l=0.1u
M13 outputp input<13> vdd vdd EEPMOS w=1u l=0.1u
M14 outputp input<14> vdd vdd EEPMOS w=1u l=0.1u
M15 outputp input<15> vdd vdd EEPMOS w=1u l=0.1u

R1 outputp vss 2k
R3 vdd outputp 2k

R2 outputn vss 1k

.ends dac
