
.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)


.subckt phase_blender ph_in[1] ph_in[0] thm_sel_bld[0] thm_sel_bld[1] thm_sel_bld[2] thm_sel_bld[3] thm_sel_bld[4] thm_sel_bld[5] thm_sel_bld[6] thm_sel_bld[7] thm_sel_bld[8] thm_sel_bld[9] thm_sel_bld[10] thm_sel_bld[11] thm_sel_bld[12] thm_sel_bld[13] thm_sel_bld[14] thm_sel_bld[15] ph_out vdd


Rtest ph_out ph_in[1] 1000



* Buffers for input phases
* buffer for ph_in[0]
MbufnA0_0 buf_inv0_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA0_0 buf_inv0_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB0_0 buf0_0 buf_inv0_0 0 0 EENMOS w=2.5u l=1u
MbufpB0_0 buf0_0 buf_inv0_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf0_0 buf0_0 0 1f
Cbuf_inv0_0 buf_inv0_0 0 1f
* buffer for ph_in[1]
MbufnA0_1 buf_inv0_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA0_1 buf_inv0_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB0_1 buf0_1 buf_inv0_1 0 0 EENMOS w=2.5u l=1u
MbufpB0_1 buf0_1 buf_inv0_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf0_1 buf0_1 0 1f
Cbuf_inv0_1 buf_inv0_1 0 1f

* buffer for ph_in[0]
MbufnA1_0 buf_inv1_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA1_0 buf_inv1_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB1_0 buf1_0 buf_inv1_0 0 0 EENMOS w=2.5u l=1u
MbufpB1_0 buf1_0 buf_inv1_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf1_0 buf1_0 0 1f
Cbuf_inv1_0 buf_inv1_0 0 1f
* buffer for ph_in[1]
MbufnA1_1 buf_inv1_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA1_1 buf_inv1_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB1_1 buf1_1 buf_inv1_1 0 0 EENMOS w=2.5u l=1u
MbufpB1_1 buf1_1 buf_inv1_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf1_1 buf1_1 0 1f
Cbuf_inv1_1 buf_inv1_1 0 1f

* buffer for ph_in[0]
MbufnA2_0 buf_inv2_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA2_0 buf_inv2_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB2_0 buf2_0 buf_inv2_0 0 0 EENMOS w=2.5u l=1u
MbufpB2_0 buf2_0 buf_inv2_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf2_0 buf2_0 0 1f
Cbuf_inv2_0 buf_inv2_0 0 1f
* buffer for ph_in[1]
MbufnA2_1 buf_inv2_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA2_1 buf_inv2_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB2_1 buf2_1 buf_inv2_1 0 0 EENMOS w=2.5u l=1u
MbufpB2_1 buf2_1 buf_inv2_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf2_1 buf2_1 0 1f
Cbuf_inv2_1 buf_inv2_1 0 1f

* buffer for ph_in[0]
MbufnA3_0 buf_inv3_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA3_0 buf_inv3_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB3_0 buf3_0 buf_inv3_0 0 0 EENMOS w=2.5u l=1u
MbufpB3_0 buf3_0 buf_inv3_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf3_0 buf3_0 0 1f
Cbuf_inv3_0 buf_inv3_0 0 1f
* buffer for ph_in[1]
MbufnA3_1 buf_inv3_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA3_1 buf_inv3_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB3_1 buf3_1 buf_inv3_1 0 0 EENMOS w=2.5u l=1u
MbufpB3_1 buf3_1 buf_inv3_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf3_1 buf3_1 0 1f
Cbuf_inv3_1 buf_inv3_1 0 1f

* buffer for ph_in[0]
MbufnA4_0 buf_inv4_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA4_0 buf_inv4_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB4_0 buf4_0 buf_inv4_0 0 0 EENMOS w=2.5u l=1u
MbufpB4_0 buf4_0 buf_inv4_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf4_0 buf4_0 0 1f
Cbuf_inv4_0 buf_inv4_0 0 1f
* buffer for ph_in[1]
MbufnA4_1 buf_inv4_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA4_1 buf_inv4_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB4_1 buf4_1 buf_inv4_1 0 0 EENMOS w=2.5u l=1u
MbufpB4_1 buf4_1 buf_inv4_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf4_1 buf4_1 0 1f
Cbuf_inv4_1 buf_inv4_1 0 1f

* buffer for ph_in[0]
MbufnA5_0 buf_inv5_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA5_0 buf_inv5_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB5_0 buf5_0 buf_inv5_0 0 0 EENMOS w=2.5u l=1u
MbufpB5_0 buf5_0 buf_inv5_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf5_0 buf5_0 0 1f
Cbuf_inv5_0 buf_inv5_0 0 1f
* buffer for ph_in[1]
MbufnA5_1 buf_inv5_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA5_1 buf_inv5_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB5_1 buf5_1 buf_inv5_1 0 0 EENMOS w=2.5u l=1u
MbufpB5_1 buf5_1 buf_inv5_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf5_1 buf5_1 0 1f
Cbuf_inv5_1 buf_inv5_1 0 1f

* buffer for ph_in[0]
MbufnA6_0 buf_inv6_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA6_0 buf_inv6_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB6_0 buf6_0 buf_inv6_0 0 0 EENMOS w=2.5u l=1u
MbufpB6_0 buf6_0 buf_inv6_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf6_0 buf6_0 0 1f
Cbuf_inv6_0 buf_inv6_0 0 1f
* buffer for ph_in[1]
MbufnA6_1 buf_inv6_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA6_1 buf_inv6_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB6_1 buf6_1 buf_inv6_1 0 0 EENMOS w=2.5u l=1u
MbufpB6_1 buf6_1 buf_inv6_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf6_1 buf6_1 0 1f
Cbuf_inv6_1 buf_inv6_1 0 1f

* buffer for ph_in[0]
MbufnA7_0 buf_inv7_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA7_0 buf_inv7_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB7_0 buf7_0 buf_inv7_0 0 0 EENMOS w=2.5u l=1u
MbufpB7_0 buf7_0 buf_inv7_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf7_0 buf7_0 0 1f
Cbuf_inv7_0 buf_inv7_0 0 1f
* buffer for ph_in[1]
MbufnA7_1 buf_inv7_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA7_1 buf_inv7_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB7_1 buf7_1 buf_inv7_1 0 0 EENMOS w=2.5u l=1u
MbufpB7_1 buf7_1 buf_inv7_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf7_1 buf7_1 0 1f
Cbuf_inv7_1 buf_inv7_1 0 1f

* buffer for ph_in[0]
MbufnA8_0 buf_inv8_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA8_0 buf_inv8_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB8_0 buf8_0 buf_inv8_0 0 0 EENMOS w=2.5u l=1u
MbufpB8_0 buf8_0 buf_inv8_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf8_0 buf8_0 0 1f
Cbuf_inv8_0 buf_inv8_0 0 1f
* buffer for ph_in[1]
MbufnA8_1 buf_inv8_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA8_1 buf_inv8_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB8_1 buf8_1 buf_inv8_1 0 0 EENMOS w=2.5u l=1u
MbufpB8_1 buf8_1 buf_inv8_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf8_1 buf8_1 0 1f
Cbuf_inv8_1 buf_inv8_1 0 1f

* buffer for ph_in[0]
MbufnA9_0 buf_inv9_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA9_0 buf_inv9_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB9_0 buf9_0 buf_inv9_0 0 0 EENMOS w=2.5u l=1u
MbufpB9_0 buf9_0 buf_inv9_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf9_0 buf9_0 0 1f
Cbuf_inv9_0 buf_inv9_0 0 1f
* buffer for ph_in[1]
MbufnA9_1 buf_inv9_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA9_1 buf_inv9_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB9_1 buf9_1 buf_inv9_1 0 0 EENMOS w=2.5u l=1u
MbufpB9_1 buf9_1 buf_inv9_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf9_1 buf9_1 0 1f
Cbuf_inv9_1 buf_inv9_1 0 1f

* buffer for ph_in[0]
MbufnA10_0 buf_inv10_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA10_0 buf_inv10_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB10_0 buf10_0 buf_inv10_0 0 0 EENMOS w=2.5u l=1u
MbufpB10_0 buf10_0 buf_inv10_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf10_0 buf10_0 0 1f
Cbuf_inv10_0 buf_inv10_0 0 1f
* buffer for ph_in[1]
MbufnA10_1 buf_inv10_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA10_1 buf_inv10_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB10_1 buf10_1 buf_inv10_1 0 0 EENMOS w=2.5u l=1u
MbufpB10_1 buf10_1 buf_inv10_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf10_1 buf10_1 0 1f
Cbuf_inv10_1 buf_inv10_1 0 1f

* buffer for ph_in[0]
MbufnA11_0 buf_inv11_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA11_0 buf_inv11_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB11_0 buf11_0 buf_inv11_0 0 0 EENMOS w=2.5u l=1u
MbufpB11_0 buf11_0 buf_inv11_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf11_0 buf11_0 0 1f
Cbuf_inv11_0 buf_inv11_0 0 1f
* buffer for ph_in[1]
MbufnA11_1 buf_inv11_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA11_1 buf_inv11_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB11_1 buf11_1 buf_inv11_1 0 0 EENMOS w=2.5u l=1u
MbufpB11_1 buf11_1 buf_inv11_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf11_1 buf11_1 0 1f
Cbuf_inv11_1 buf_inv11_1 0 1f

* buffer for ph_in[0]
MbufnA12_0 buf_inv12_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA12_0 buf_inv12_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB12_0 buf12_0 buf_inv12_0 0 0 EENMOS w=2.5u l=1u
MbufpB12_0 buf12_0 buf_inv12_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf12_0 buf12_0 0 1f
Cbuf_inv12_0 buf_inv12_0 0 1f
* buffer for ph_in[1]
MbufnA12_1 buf_inv12_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA12_1 buf_inv12_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB12_1 buf12_1 buf_inv12_1 0 0 EENMOS w=2.5u l=1u
MbufpB12_1 buf12_1 buf_inv12_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf12_1 buf12_1 0 1f
Cbuf_inv12_1 buf_inv12_1 0 1f

* buffer for ph_in[0]
MbufnA13_0 buf_inv13_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA13_0 buf_inv13_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB13_0 buf13_0 buf_inv13_0 0 0 EENMOS w=2.5u l=1u
MbufpB13_0 buf13_0 buf_inv13_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf13_0 buf13_0 0 1f
Cbuf_inv13_0 buf_inv13_0 0 1f
* buffer for ph_in[1]
MbufnA13_1 buf_inv13_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA13_1 buf_inv13_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB13_1 buf13_1 buf_inv13_1 0 0 EENMOS w=2.5u l=1u
MbufpB13_1 buf13_1 buf_inv13_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf13_1 buf13_1 0 1f
Cbuf_inv13_1 buf_inv13_1 0 1f

* buffer for ph_in[0]
MbufnA14_0 buf_inv14_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA14_0 buf_inv14_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB14_0 buf14_0 buf_inv14_0 0 0 EENMOS w=2.5u l=1u
MbufpB14_0 buf14_0 buf_inv14_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf14_0 buf14_0 0 1f
Cbuf_inv14_0 buf_inv14_0 0 1f
* buffer for ph_in[1]
MbufnA14_1 buf_inv14_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA14_1 buf_inv14_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB14_1 buf14_1 buf_inv14_1 0 0 EENMOS w=2.5u l=1u
MbufpB14_1 buf14_1 buf_inv14_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf14_1 buf14_1 0 1f
Cbuf_inv14_1 buf_inv14_1 0 1f

* buffer for ph_in[0]
MbufnA15_0 buf_inv15_0 ph_in[0] 0 0 EENMOS w=0.25u l=1u
MbufpA15_0 buf_inv15_0 ph_in[0] vdd vdd EEPMOS w=0.5u l=1u
MbufnB15_0 buf15_0 buf_inv15_0 0 0 EENMOS w=2.5u l=1u
MbufpB15_0 buf15_0 buf_inv15_0 vdd vdd EEPMOS w=5.0u l=1u
Cbuf15_0 buf15_0 0 1f
Cbuf_inv15_0 buf_inv15_0 0 1f
* buffer for ph_in[1]
MbufnA15_1 buf_inv15_1 ph_in[1] 0 0 EENMOS w=0.25u l=1u
MbufpA15_1 buf_inv15_1 ph_in[1] vdd vdd EEPMOS w=0.5u l=1u
MbufnB15_1 buf15_1 buf_inv15_1 0 0 EENMOS w=2.5u l=1u
MbufpB15_1 buf15_1 buf_inv15_1 vdd vdd EEPMOS w=5.0u l=1u
Cbuf15_1 buf15_1 0 1f
Cbuf_inv15_1 buf_inv15_1 0 1f


* Inverters for clocks
* inv for thm_sel_bld[0]
Minvn0 clk_0_inv thm_sel_bld[0] 0 0 EENMOS w=0.25u l=1u
Minvp0 clk_0_inv thm_sel_bld[0] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[1]
Minvn1 clk_1_inv thm_sel_bld[1] 0 0 EENMOS w=0.25u l=1u
Minvp1 clk_1_inv thm_sel_bld[1] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[2]
Minvn2 clk_2_inv thm_sel_bld[2] 0 0 EENMOS w=0.25u l=1u
Minvp2 clk_2_inv thm_sel_bld[2] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[3]
Minvn3 clk_3_inv thm_sel_bld[3] 0 0 EENMOS w=0.25u l=1u
Minvp3 clk_3_inv thm_sel_bld[3] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[4]
Minvn4 clk_4_inv thm_sel_bld[4] 0 0 EENMOS w=0.25u l=1u
Minvp4 clk_4_inv thm_sel_bld[4] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[5]
Minvn5 clk_5_inv thm_sel_bld[5] 0 0 EENMOS w=0.25u l=1u
Minvp5 clk_5_inv thm_sel_bld[5] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[6]
Minvn6 clk_6_inv thm_sel_bld[6] 0 0 EENMOS w=0.25u l=1u
Minvp6 clk_6_inv thm_sel_bld[6] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[7]
Minvn7 clk_7_inv thm_sel_bld[7] 0 0 EENMOS w=0.25u l=1u
Minvp7 clk_7_inv thm_sel_bld[7] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[8]
Minvn8 clk_8_inv thm_sel_bld[8] 0 0 EENMOS w=0.25u l=1u
Minvp8 clk_8_inv thm_sel_bld[8] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[9]
Minvn9 clk_9_inv thm_sel_bld[9] 0 0 EENMOS w=0.25u l=1u
Minvp9 clk_9_inv thm_sel_bld[9] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[10]
Minvn10 clk_10_inv thm_sel_bld[10] 0 0 EENMOS w=0.25u l=1u
Minvp10 clk_10_inv thm_sel_bld[10] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[11]
Minvn11 clk_11_inv thm_sel_bld[11] 0 0 EENMOS w=0.25u l=1u
Minvp11 clk_11_inv thm_sel_bld[11] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[12]
Minvn12 clk_12_inv thm_sel_bld[12] 0 0 EENMOS w=0.25u l=1u
Minvp12 clk_12_inv thm_sel_bld[12] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[13]
Minvn13 clk_13_inv thm_sel_bld[13] 0 0 EENMOS w=0.25u l=1u
Minvp13 clk_13_inv thm_sel_bld[13] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[14]
Minvn14 clk_14_inv thm_sel_bld[14] 0 0 EENMOS w=0.25u l=1u
Minvp14 clk_14_inv thm_sel_bld[14] vdd vdd EEPMOS w=0.5u l=1u

* inv for thm_sel_bld[15]
Minvn15 clk_15_inv thm_sel_bld[15] 0 0 EENMOS w=0.25u l=1u
Minvp15 clk_15_inv thm_sel_bld[15] vdd vdd EEPMOS w=0.5u l=1u


* Muxes
* mux for ph_in[0], thm_sel_bld[0]
Mmuxn0_0 buf0_0 clk_0_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp0_0 buf0_0 thm_sel_bld[0] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[0]
Mmuxn0_1 buf0_1 thm_sel_bld[0] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp0_1 buf0_1 clk_0_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[1]
Mmuxn1_0 buf1_0 clk_1_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp1_0 buf1_0 thm_sel_bld[1] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[1]
Mmuxn1_1 buf1_1 thm_sel_bld[1] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp1_1 buf1_1 clk_1_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[2]
Mmuxn2_0 buf2_0 clk_2_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp2_0 buf2_0 thm_sel_bld[2] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[2]
Mmuxn2_1 buf2_1 thm_sel_bld[2] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp2_1 buf2_1 clk_2_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[3]
Mmuxn3_0 buf3_0 clk_3_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp3_0 buf3_0 thm_sel_bld[3] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[3]
Mmuxn3_1 buf3_1 thm_sel_bld[3] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp3_1 buf3_1 clk_3_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[4]
Mmuxn4_0 buf4_0 clk_4_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp4_0 buf4_0 thm_sel_bld[4] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[4]
Mmuxn4_1 buf4_1 thm_sel_bld[4] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp4_1 buf4_1 clk_4_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[5]
Mmuxn5_0 buf5_0 clk_5_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp5_0 buf5_0 thm_sel_bld[5] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[5]
Mmuxn5_1 buf5_1 thm_sel_bld[5] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp5_1 buf5_1 clk_5_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[6]
Mmuxn6_0 buf6_0 clk_6_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp6_0 buf6_0 thm_sel_bld[6] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[6]
Mmuxn6_1 buf6_1 thm_sel_bld[6] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp6_1 buf6_1 clk_6_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[7]
Mmuxn7_0 buf7_0 clk_7_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp7_0 buf7_0 thm_sel_bld[7] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[7]
Mmuxn7_1 buf7_1 thm_sel_bld[7] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp7_1 buf7_1 clk_7_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[8]
Mmuxn8_0 buf8_0 clk_8_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp8_0 buf8_0 thm_sel_bld[8] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[8]
Mmuxn8_1 buf8_1 thm_sel_bld[8] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp8_1 buf8_1 clk_8_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[9]
Mmuxn9_0 buf9_0 clk_9_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp9_0 buf9_0 thm_sel_bld[9] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[9]
Mmuxn9_1 buf9_1 thm_sel_bld[9] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp9_1 buf9_1 clk_9_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[10]
Mmuxn10_0 buf10_0 clk_10_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp10_0 buf10_0 thm_sel_bld[10] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[10]
Mmuxn10_1 buf10_1 thm_sel_bld[10] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp10_1 buf10_1 clk_10_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[11]
Mmuxn11_0 buf11_0 clk_11_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp11_0 buf11_0 thm_sel_bld[11] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[11]
Mmuxn11_1 buf11_1 thm_sel_bld[11] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp11_1 buf11_1 clk_11_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[12]
Mmuxn12_0 buf12_0 clk_12_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp12_0 buf12_0 thm_sel_bld[12] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[12]
Mmuxn12_1 buf12_1 thm_sel_bld[12] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp12_1 buf12_1 clk_12_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[13]
Mmuxn13_0 buf13_0 clk_13_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp13_0 buf13_0 thm_sel_bld[13] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[13]
Mmuxn13_1 buf13_1 thm_sel_bld[13] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp13_1 buf13_1 clk_13_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[14]
Mmuxn14_0 buf14_0 clk_14_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp14_0 buf14_0 thm_sel_bld[14] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[14]
Mmuxn14_1 buf14_1 thm_sel_bld[14] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp14_1 buf14_1 clk_14_inv ph_out vdd EEPMOS w=0.5u l=1u

* mux for ph_in[0], thm_sel_bld[15]
Mmuxn15_0 buf15_0 clk_15_inv ph_out 0 EENMOS w=0.25u l=1u
Mmuxp15_0 buf15_0 thm_sel_bld[15] ph_out vdd EEPMOS w=0.5u l=1u
* mux for ph_in[1], thm_sel_bld[15]
Mmuxn15_1 buf15_1 thm_sel_bld[15] ph_out 0 EENMOS w=0.25u l=1u
Mmuxp15_1 buf15_1 clk_15_inv ph_out vdd EEPMOS w=0.5u l=1u


.ends
