
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt sampler5 Vcal Vin clk_v2t clk_v2t_e clk_v2t_eb clk_v2t_gated clk_v2t_l clk_v2t_lb clk_v2tb clk_v2tb_gated ctl<0> ctl<10> ctl<11> ctl<12> ctl<13> ctl<14> ctl<15> ctl<16> ctl<17> ctl<18> ctl<19> ctl<1> ctl<20> ctl<21> ctl<22> ctl<23> ctl<24> ctl<25> ctl<26> ctl<27> ctl<28> ctl<29> ctl<2> ctl<30> ctl<31> ctl<3> ctl<4> ctl<5> ctl<6> ctl<7> ctl<8> ctl<9> v2t_out vdd

* sampling NMOS
Msamp discharge clk_v2t Vin 0 EENMOS w=1u l=0.1u

* discharge RC
Rdischarge discharge 0 10000000
Cdischarge discharge 0 10p

* trigger inverter
Minvn 0   discharge trigger 0   EENMOS w=1u   l=0.1u
Minvp vdd discharge trigger vdd EEPMOS w=0.5u l=0.1u

* output NOR
Mnorn1 0 trigger  v2t_out 0 EENMOS w=1u l=0.1u
Mnorn2 0 clk_v2t v2t_out 0 EENMOS w=1u l=0.1u
Mnorp1 vdd trigger nor_temp vdd    EEPMOS w=1u l=0.1u
Mnorp2 nor_temp    clk_v2t v2t_out vdd EEPMOS w=1u l=0.1u

* output cap
Cout v2t_out 0 1f

* ignore caps
Cignore_0 Vcal 0 1f
Cignore_1 clk_v2t_e 0 1f
Cignore_2 clk_v2t_eb 0 1f
Cignore_3 clk_v2t_gated 0 1f
Cignore_4 clk_v2t_l 0 1f
Cignore_5 clk_v2t_lb 0 1f
Cignore_6 clk_v2tb 0 1f
Cignore_7 clk_v2tb_gated 0 1f
Cignore_8 ctl<0> 0 1f
Cignore_9 ctl<10> 0 1f
Cignore_10 ctl<11> 0 1f
Cignore_11 ctl<12> 0 1f
Cignore_12 ctl<13> 0 1f
Cignore_13 ctl<14> 0 1f
Cignore_14 ctl<15> 0 1f
Cignore_15 ctl<16> 0 1f
Cignore_16 ctl<17> 0 1f
Cignore_17 ctl<18> 0 1f
Cignore_18 ctl<19> 0 1f
Cignore_19 ctl<1> 0 1f
Cignore_20 ctl<20> 0 1f
Cignore_21 ctl<21> 0 1f
Cignore_22 ctl<22> 0 1f
Cignore_23 ctl<23> 0 1f
Cignore_24 ctl<24> 0 1f
Cignore_25 ctl<25> 0 1f
Cignore_26 ctl<26> 0 1f
Cignore_27 ctl<27> 0 1f
Cignore_28 ctl<28> 0 1f
Cignore_29 ctl<29> 0 1f
Cignore_30 ctl<2> 0 1f
Cignore_31 ctl<30> 0 1f
Cignore_32 ctl<31> 0 1f
Cignore_33 ctl<3> 0 1f
Cignore_34 ctl<4> 0 1f
Cignore_35 ctl<5> 0 1f
Cignore_36 ctl<6> 0 1f
Cignore_37 ctl<7> 0 1f
Cignore_38 ctl<8> 0 1f
Cignore_39 ctl<9> 0 1f

* add some effect for late clock
Cslow slow 0 5p
Mslow slow clk_v2t_l discharge 0 EENMOS w=1u l=0.1u

* debug
*Rdebug discharge_0 z_debug 0

.ends sampler4
