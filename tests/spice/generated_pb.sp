.subckt phase_blender ph_in[1] ph_in[0] thm_sel_bld[0] thm_sel_bld[1] thm_sel_bld[2] thm_sel_bld[3] ph_out vdd
* Inverters for clocks
* inv for thm_sel_bld[0]
Minvn0 clk_0_inv thm_sel_bld[0] 0 0 EENMOS w=.1u l=.1u
Minvp0 clk_0_inv thm_sel_bld[0] vdd vdd EEPMOS w=.1u l=.1u

* inv for thm_sel_bld[1]
Minvn1 clk_1_inv thm_sel_bld[1] 0 0 EENMOS w=.1u l=.1u
Minvp1 clk_1_inv thm_sel_bld[1] vdd vdd EEPMOS w=.1u l=.1u

* inv for thm_sel_bld[2]
Minvn2 clk_2_inv thm_sel_bld[2] 0 0 EENMOS w=.1u l=.1u
Minvp2 clk_2_inv thm_sel_bld[2] vdd vdd EEPMOS w=.1u l=.1u

* inv for thm_sel_bld[3]
Minvn3 clk_3_inv thm_sel_bld[3] 0 0 EENMOS w=.1u l=.1u
Minvp3 clk_3_inv thm_sel_bld[3] vdd vdd EEPMOS w=.1u l=.1u

* Muxes
* mux for ph_in[0], thm_sel_bld[0]
Mmuxn0_0 ph_in[0] clk_0_inv ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp0_0 ph_in[0] thm_sel_bld[0] ph_out vdd EENMOS w=0.2 l=.1u
* mux for ph_in[1], thm_sel_bld[0]
Mmuxn0_1 ph_in[1] thm_sel_bld[0] ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp0_1 ph_in[1] clk_0_inv ph_out vdd EENMOS w=0.2 l=.1u

* mux for ph_in[0], thm_sel_bld[1]
Mmuxn1_0 ph_in[0] clk_1_inv ph_out 0 EENMOS w=0.2 l=.1u
Mmuxp1_0 ph_in[0] thm_sel_bld[1] ph_out vdd EENMOS w=0.4 l=.1u
* mux for ph_in[1], thm_sel_bld[1]
Mmuxn1_1 ph_in[1] thm_sel_bld[1] ph_out 0 EENMOS w=0.2 l=.1u
Mmuxp1_1 ph_in[1] clk_1_inv ph_out vdd EENMOS w=0.4 l=.1u

* mux for ph_in[0], thm_sel_bld[2]
Mmuxn2_0 ph_in[0] clk_2_inv ph_out 0 EENMOS w=0.4 l=.1u
Mmuxp2_0 ph_in[0] thm_sel_bld[2] ph_out vdd EENMOS w=0.8 l=.1u
* mux for ph_in[1], thm_sel_bld[2]
Mmuxn2_1 ph_in[1] thm_sel_bld[2] ph_out 0 EENMOS w=0.4 l=.1u
Mmuxp2_1 ph_in[1] clk_2_inv ph_out vdd EENMOS w=0.8 l=.1u

* mux for ph_in[0], thm_sel_bld[3]
Mmuxn3_0 ph_in[0] clk_3_inv ph_out 0 EENMOS w=0.8 l=.1u
Mmuxp3_0 ph_in[0] thm_sel_bld[3] ph_out vdd EENMOS w=1.6 l=.1u
* mux for ph_in[1], thm_sel_bld[3]
Mmuxn3_1 ph_in[1] thm_sel_bld[3] ph_out 0 EENMOS w=0.8 l=.1u
Mmuxp3_1 ph_in[1] clk_3_inv ph_out vdd EENMOS w=1.6 l=.1u


.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.ends
