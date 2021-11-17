
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt sampler7 clkn<0> clkn<1> clkn<2> clkn<3> clkp<0> clkp<1> clkp<2> clkp<3> in_ out<0> out<1> out<2> out<3> vdd vssi debug

* Mock 50 Ohm input driver
Rdriver in_ in_nonideal 50

* sampling NMOS
Msampn0 out<0> clkn<0> in_nonideal vss EENMOS w=10u l=0.1u
Cgsn0 clkn<0> out<0> 10f
Cgdn0 clkn<0> in_nonideal 10f
Cdsn0 in_nonideal out<0> 10f
Msampn1 out<1> clkn<1> in_nonideal vss EENMOS w=10u l=0.1u
Cgsn1 clkn<1> out<1> 10f
Cgdn1 clkn<1> in_nonideal 10f
Cdsn1 in_nonideal out<1> 10f
Msampn2 out<2> clkn<2> in_nonideal vss EENMOS w=10u l=0.1u
Cgsn2 clkn<2> out<2> 10f
Cgdn2 clkn<2> in_nonideal 10f
Cdsn2 in_nonideal out<2> 10f
Msampn3 out<3> clkn<3> in_nonideal vss EENMOS w=10u l=0.1u
Cgsn3 clkn<3> out<3> 10f
Cgdn3 clkn<3> in_nonideal 10f
Cdsn3 in_nonideal out<3> 10f

* sampling PMOS
Msampp0 out<0> clkp<0> in_nonideal vdd EEPMOS w=20u l=0.1u
Cgsp0 clkp<0> out<0> 10f
Cgdp0 clkp<0> in_nonideal 10f
Cdsp0 in_nonideal out<0> 10f
Msampp1 out<1> clkp<1> in_nonideal vdd EEPMOS w=20u l=0.1u
Cgsp1 clkp<1> out<1> 10f
Cgdp1 clkp<1> in_nonideal 10f
Cdsp1 in_nonideal out<1> 10f
Msampp2 out<2> clkp<2> in_nonideal vdd EEPMOS w=20u l=0.1u
Cgsp2 clkp<2> out<2> 10f
Cgdp2 clkp<2> in_nonideal 10f
Cdsp2 in_nonideal out<2> 10f
Msampp3 out<3> clkp<3> in_nonideal vdd EEPMOS w=20u l=0.1u
Cgsp3 clkp<3> out<3> 10f
Cgdp3 clkp<3> in_nonideal 10f
Cdsp3 in_nonideal out<3> 10f

* output cap
Cout0 out<0> 0 50000f
Cout1 out<1> 0 50000f
Cout2 out<2> 0 50000f
Cout3 out<3> 0 50000f

* debug
Rdebug debug in_nonideal 0

.ends sampler7
