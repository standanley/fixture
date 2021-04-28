* NMOS/PMOS models from
* https://people.rit.edu/lffeee/SPICE_Examples.pdf

.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt inv in out vdd vss vtop vbottom
Mpmos vtop in out vdd EEPMOS w=1u l=0.1u
Mnmos vbottom in out vss EENMOS w=1u l=0.1u
.ends inv

.subckt dac input<0> input<1> outputp outputn vdd vss

ibias vbias 0 10u
Msetbias vbias vbias vdd vdd EEPMOS w=1u l=0.1u


Xctrl0 input<0> ctrl0 vdd vss vdd vbias inv
M0 vdd ctrl0 outputp vdd EEPMOS w=1u l=0.1u

Xctrl1 input<1> ctrl1 vdd vss vdd vbias inv
M1 vdd ctrl1 outputp vdd EEPMOS w=2u l=0.1u

* R = (vdd/2) / ((2^N-1) * ibias)
R1 outputp vss 30k

* not using outputn for now
R2 outputn vss 1k

.ends dac
