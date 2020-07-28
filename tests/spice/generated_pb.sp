.subckt phase_blender ph_in[1] ph_in[0] thm_sel_bld[0] thm_sel_bld[1] thm_sel_bld[2] thm_sel_bld[3] ph_out vdd
* Buffers for input phases
* buffer for ph_in[0]
MbufnA0_0 buf_inv0_0 ph_in[0] 0 0 EENMOS w=.1u l=1u
MbufpA0_0 buf_inv0_0 ph_in[0] vdd vdd EEPMOS w=.2u l=1u
MbufnB0_0 buf0_0 buf_inv0_0 0 0 EENMOS w=.01u l=1u
MbufpB0_0 buf0_0 buf_inv0_0 vdd vdd EEPMOS w=.02u l=1u
* buffer for ph_in[1]
MbufnA0_1 buf_inv0_1 ph_in[1] 0 0 EENMOS w=.1u l=1u
MbufpA0_1 buf_inv0_1 ph_in[1] vdd vdd EEPMOS w=.2u l=1u
MbufnB0_1 buf0_1 buf_inv0_1 0 0 EENMOS w=.01u l=1u
MbufpB0_1 buf0_1 buf_inv0_1 vdd vdd EEPMOS w=.02u l=1u

* buffer for ph_in[0]
MbufnA1_0 buf_inv1_0 ph_in[0] 0 0 EENMOS w=.1u l=1u
MbufpA1_0 buf_inv1_0 ph_in[0] vdd vdd EEPMOS w=.2u l=1u
MbufnB1_0 buf1_0 buf_inv1_0 0 0 EENMOS w=.01u l=1u
MbufpB1_0 buf1_0 buf_inv1_0 vdd vdd EEPMOS w=.02u l=1u
* buffer for ph_in[1]
MbufnA1_1 buf_inv1_1 ph_in[1] 0 0 EENMOS w=.1u l=1u
MbufpA1_1 buf_inv1_1 ph_in[1] vdd vdd EEPMOS w=.2u l=1u
MbufnB1_1 buf1_1 buf_inv1_1 0 0 EENMOS w=.01u l=1u
MbufpB1_1 buf1_1 buf_inv1_1 vdd vdd EEPMOS w=.02u l=1u

* buffer for ph_in[0]
MbufnA2_0 buf_inv2_0 ph_in[0] 0 0 EENMOS w=.1u l=1u
MbufpA2_0 buf_inv2_0 ph_in[0] vdd vdd EEPMOS w=.2u l=1u
MbufnB2_0 buf2_0 buf_inv2_0 0 0 EENMOS w=.01u l=1u
MbufpB2_0 buf2_0 buf_inv2_0 vdd vdd EEPMOS w=.02u l=1u
* buffer for ph_in[1]
MbufnA2_1 buf_inv2_1 ph_in[1] 0 0 EENMOS w=.1u l=1u
MbufpA2_1 buf_inv2_1 ph_in[1] vdd vdd EEPMOS w=.2u l=1u
MbufnB2_1 buf2_1 buf_inv2_1 0 0 EENMOS w=.01u l=1u
MbufpB2_1 buf2_1 buf_inv2_1 vdd vdd EEPMOS w=.02u l=1u

* buffer for ph_in[0]
MbufnA3_0 buf_inv3_0 ph_in[0] 0 0 EENMOS w=.1u l=1u
MbufpA3_0 buf_inv3_0 ph_in[0] vdd vdd EEPMOS w=.2u l=1u
MbufnB3_0 buf3_0 buf_inv3_0 0 0 EENMOS w=.01u l=1u
MbufpB3_0 buf3_0 buf_inv3_0 vdd vdd EEPMOS w=.02u l=1u
* buffer for ph_in[1]
MbufnA3_1 buf_inv3_1 ph_in[1] 0 0 EENMOS w=.1u l=1u
MbufpA3_1 buf_inv3_1 ph_in[1] vdd vdd EEPMOS w=.2u l=1u
MbufnB3_1 buf3_1 buf_inv3_1 0 0 EENMOS w=.01u l=1u
MbufpB3_1 buf3_1 buf_inv3_1 vdd vdd EEPMOS w=.02u l=1u

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
Mmuxn0_0 buf0_0 clk_0_inv ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp0_0 buf0_0 thm_sel_bld[0] ph_out vdd EEPMOS w=0.2 l=.1u
* mux for ph_in[1], thm_sel_bld[0]
Mmuxn0_1 buf0_1 thm_sel_bld[0] ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp0_1 buf0_1 clk_0_inv ph_out vdd EEPMOS w=0.2 l=.1u

* mux for ph_in[0], thm_sel_bld[1]
Mmuxn1_0 buf1_0 clk_1_inv ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp1_0 buf1_0 thm_sel_bld[1] ph_out vdd EEPMOS w=0.2 l=.1u
* mux for ph_in[1], thm_sel_bld[1]
Mmuxn1_1 buf1_1 thm_sel_bld[1] ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp1_1 buf1_1 clk_1_inv ph_out vdd EEPMOS w=0.2 l=.1u

* mux for ph_in[0], thm_sel_bld[2]
Mmuxn2_0 buf2_0 clk_2_inv ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp2_0 buf2_0 thm_sel_bld[2] ph_out vdd EEPMOS w=0.2 l=.1u
* mux for ph_in[1], thm_sel_bld[2]
Mmuxn2_1 buf2_1 thm_sel_bld[2] ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp2_1 buf2_1 clk_2_inv ph_out vdd EEPMOS w=0.2 l=.1u

* mux for ph_in[0], thm_sel_bld[3]
Mmuxn3_0 buf3_0 clk_3_inv ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp3_0 buf3_0 thm_sel_bld[3] ph_out vdd EEPMOS w=0.2 l=.1u
* mux for ph_in[1], thm_sel_bld[3]
Mmuxn3_1 buf3_1 thm_sel_bld[3] ph_out 0 EENMOS w=0.1 l=.1u
Mmuxp3_1 buf3_1 clk_3_inv ph_out vdd EEPMOS w=0.2 l=.1u

Ctest ph_out 0 0.1f


.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.ends
