
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt sampler4 clk_v2t<0> clk_v2t<1> clk_v2t_e<0> clk_v2t_e<1> clk_v2t_eb<0> clk_v2t_eb<1> clk_v2t_gated<0> clk_v2t_gated<1> clk_v2t_l<0> clk_v2t_l<1> clk_v2t_lb<0> clk_v2t_lb<1> clk_v2tb<0> clk_v2tb<1> clk_v2tb_gated<0> clk_v2tb_gated<1> in_ out<0> out<1> vdd z_debug

* sampling NMOS
Msamp0 discharge_0 clk_v2t<0> in_ 0 EENMOS w=1u l=0.1u
Msamp1 discharge_1 clk_v2t<1> in_ 0 EENMOS w=1u l=0.1u

* discharge RC
Rdischarge0 discharge_0 0 10000000
Cdischarge0 discharge_0 0 10p
Rdischarge1 discharge_1 0 10000000
Cdischarge1 discharge_1 0 10p

* trigger inverter
Minvn0 0   discharge_0 trigger_0 0   EENMOS w=1u   l=0.1u
Minvp0 vdd discharge_0 trigger_0 vdd EEPMOS w=0.5u l=0.1u
Minvn1 0   discharge_1 trigger_1 0   EENMOS w=1u   l=0.1u
Minvp1 vdd discharge_1 trigger_1 vdd EEPMOS w=0.5u l=0.1u

* output NOR
Mnorn10 0 trigger_0  out<0> 0 EENMOS w=1u l=0.1u
Mnorn20 0 clk_v2t<0> out<0> 0 EENMOS w=1u l=0.1u
Mnorp10 vdd trigger_0 nor_temp_0 vdd    EEPMOS w=1u l=0.1u
Mnorp20 nor_temp_0    clk_v2t<0> out<0> vdd EEPMOS w=1u l=0.1u
Mnorn11 0 trigger_1  out<1> 0 EENMOS w=1u l=0.1u
Mnorn21 0 clk_v2t<1> out<1> 0 EENMOS w=1u l=0.1u
Mnorp11 vdd trigger_1 nor_temp_1 vdd    EEPMOS w=1u l=0.1u
Mnorp21 nor_temp_1    clk_v2t<1> out<1> vdd EEPMOS w=1u l=0.1u

* output cap
Cout0 out<0> 0 1f
Cout1 out<1> 0 1f

* ignore caps
Cignore0 clk_v2t_e<0> 0 1f
Cignore1 clk_v2t_e<1> 0 1f
Cignore2 clk_v2t_eb<0> 0 1f
Cignore3 clk_v2t_eb<1> 0 1f
Cignore4 clk_v2t_gated<0> 0 1f
Cignore5 clk_v2t_gated<1> 0 1f
Cignore6 clk_v2t_l<0> 0 1f
Cignore7 clk_v2t_l<1> 0 1f
Cignore8 clk_v2t_lb<0> 0 1f
Cignore9 clk_v2t_lb<1> 0 1f
Cignore10 clk_v2tb<0> 0 1f
Cignore11 clk_v2tb<1> 0 1f
Cignore12 clk_v2tb_gated<0> 0 1f
Cignore13 clk_v2tb_gated<1> 0 1f
Cignore14 vdd 0 1f
Cignore15 z_debug 0 1f

* debug
Rdebug discharge_0 z_debug 0

.ends sampler4
