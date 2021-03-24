.lib "../sky130/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice" tt

.subckt phase_blender ph_in[1] ph_in[0] thm_sel_bld[0] thm_sel_bld[1] thm_sel_bld[2] thm_sel_bld[3] thm_sel_bld[4] thm_sel_bld[5] thm_sel_bld[6] thm_sel_bld[7] thm_sel_bld[8] thm_sel_bld[9] thm_sel_bld[10] thm_sel_bld[11] thm_sel_bld[12] thm_sel_bld[13] thm_sel_bld[14] thm_sel_bld[15] ph_out vdd



* Buffers for input phases
* buffer for ph_in[0]
XbufnA0_0 buf_inv0_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA0_0 buf_inv0_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB0_0 buf0_0 buf_inv0_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB0_0 buf0_0 buf_inv0_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf0_0 buf0_0 0 100f
* buffer for ph_in[1]
XbufnA0_1 buf_inv0_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA0_1 buf_inv0_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB0_1 buf0_1 buf_inv0_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB0_1 buf0_1 buf_inv0_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf0_1 buf0_1 0 100f

* buffer for ph_in[0]
XbufnA1_0 buf_inv1_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA1_0 buf_inv1_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB1_0 buf1_0 buf_inv1_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB1_0 buf1_0 buf_inv1_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf1_0 buf1_0 0 100f
* buffer for ph_in[1]
XbufnA1_1 buf_inv1_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA1_1 buf_inv1_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB1_1 buf1_1 buf_inv1_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB1_1 buf1_1 buf_inv1_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf1_1 buf1_1 0 100f

* buffer for ph_in[0]
XbufnA2_0 buf_inv2_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA2_0 buf_inv2_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB2_0 buf2_0 buf_inv2_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB2_0 buf2_0 buf_inv2_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf2_0 buf2_0 0 100f
* buffer for ph_in[1]
XbufnA2_1 buf_inv2_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA2_1 buf_inv2_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB2_1 buf2_1 buf_inv2_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB2_1 buf2_1 buf_inv2_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf2_1 buf2_1 0 100f

* buffer for ph_in[0]
XbufnA3_0 buf_inv3_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA3_0 buf_inv3_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB3_0 buf3_0 buf_inv3_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB3_0 buf3_0 buf_inv3_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf3_0 buf3_0 0 100f
* buffer for ph_in[1]
XbufnA3_1 buf_inv3_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA3_1 buf_inv3_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB3_1 buf3_1 buf_inv3_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB3_1 buf3_1 buf_inv3_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf3_1 buf3_1 0 100f

* buffer for ph_in[0]
XbufnA4_0 buf_inv4_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA4_0 buf_inv4_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB4_0 buf4_0 buf_inv4_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB4_0 buf4_0 buf_inv4_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf4_0 buf4_0 0 100f
* buffer for ph_in[1]
XbufnA4_1 buf_inv4_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA4_1 buf_inv4_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB4_1 buf4_1 buf_inv4_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB4_1 buf4_1 buf_inv4_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf4_1 buf4_1 0 100f

* buffer for ph_in[0]
XbufnA5_0 buf_inv5_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA5_0 buf_inv5_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB5_0 buf5_0 buf_inv5_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB5_0 buf5_0 buf_inv5_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf5_0 buf5_0 0 100f
* buffer for ph_in[1]
XbufnA5_1 buf_inv5_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA5_1 buf_inv5_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB5_1 buf5_1 buf_inv5_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB5_1 buf5_1 buf_inv5_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf5_1 buf5_1 0 100f

* buffer for ph_in[0]
XbufnA6_0 buf_inv6_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA6_0 buf_inv6_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB6_0 buf6_0 buf_inv6_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB6_0 buf6_0 buf_inv6_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf6_0 buf6_0 0 100f
* buffer for ph_in[1]
XbufnA6_1 buf_inv6_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA6_1 buf_inv6_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB6_1 buf6_1 buf_inv6_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB6_1 buf6_1 buf_inv6_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf6_1 buf6_1 0 100f

* buffer for ph_in[0]
XbufnA7_0 buf_inv7_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA7_0 buf_inv7_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB7_0 buf7_0 buf_inv7_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB7_0 buf7_0 buf_inv7_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf7_0 buf7_0 0 100f
* buffer for ph_in[1]
XbufnA7_1 buf_inv7_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA7_1 buf_inv7_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB7_1 buf7_1 buf_inv7_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB7_1 buf7_1 buf_inv7_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf7_1 buf7_1 0 100f

* buffer for ph_in[0]
XbufnA8_0 buf_inv8_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA8_0 buf_inv8_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB8_0 buf8_0 buf_inv8_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB8_0 buf8_0 buf_inv8_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf8_0 buf8_0 0 100f
* buffer for ph_in[1]
XbufnA8_1 buf_inv8_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA8_1 buf_inv8_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB8_1 buf8_1 buf_inv8_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB8_1 buf8_1 buf_inv8_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf8_1 buf8_1 0 100f

* buffer for ph_in[0]
XbufnA9_0 buf_inv9_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA9_0 buf_inv9_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB9_0 buf9_0 buf_inv9_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB9_0 buf9_0 buf_inv9_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf9_0 buf9_0 0 100f
* buffer for ph_in[1]
XbufnA9_1 buf_inv9_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA9_1 buf_inv9_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB9_1 buf9_1 buf_inv9_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB9_1 buf9_1 buf_inv9_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf9_1 buf9_1 0 100f

* buffer for ph_in[0]
XbufnA10_0 buf_inv10_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA10_0 buf_inv10_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB10_0 buf10_0 buf_inv10_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB10_0 buf10_0 buf_inv10_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf10_0 buf10_0 0 100f
* buffer for ph_in[1]
XbufnA10_1 buf_inv10_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA10_1 buf_inv10_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB10_1 buf10_1 buf_inv10_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB10_1 buf10_1 buf_inv10_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf10_1 buf10_1 0 100f

* buffer for ph_in[0]
XbufnA11_0 buf_inv11_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA11_0 buf_inv11_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB11_0 buf11_0 buf_inv11_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB11_0 buf11_0 buf_inv11_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf11_0 buf11_0 0 100f
* buffer for ph_in[1]
XbufnA11_1 buf_inv11_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA11_1 buf_inv11_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB11_1 buf11_1 buf_inv11_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB11_1 buf11_1 buf_inv11_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf11_1 buf11_1 0 100f

* buffer for ph_in[0]
XbufnA12_0 buf_inv12_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA12_0 buf_inv12_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB12_0 buf12_0 buf_inv12_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB12_0 buf12_0 buf_inv12_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf12_0 buf12_0 0 100f
* buffer for ph_in[1]
XbufnA12_1 buf_inv12_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA12_1 buf_inv12_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB12_1 buf12_1 buf_inv12_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB12_1 buf12_1 buf_inv12_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf12_1 buf12_1 0 100f

* buffer for ph_in[0]
XbufnA13_0 buf_inv13_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA13_0 buf_inv13_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB13_0 buf13_0 buf_inv13_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB13_0 buf13_0 buf_inv13_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf13_0 buf13_0 0 100f
* buffer for ph_in[1]
XbufnA13_1 buf_inv13_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA13_1 buf_inv13_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB13_1 buf13_1 buf_inv13_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB13_1 buf13_1 buf_inv13_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf13_1 buf13_1 0 100f

* buffer for ph_in[0]
XbufnA14_0 buf_inv14_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA14_0 buf_inv14_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB14_0 buf14_0 buf_inv14_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB14_0 buf14_0 buf_inv14_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf14_0 buf14_0 0 100f
* buffer for ph_in[1]
XbufnA14_1 buf_inv14_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA14_1 buf_inv14_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB14_1 buf14_1 buf_inv14_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB14_1 buf14_1 buf_inv14_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf14_1 buf14_1 0 100f

* buffer for ph_in[0]
XbufnA15_0 buf_inv15_0 ph_in[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA15_0 buf_inv15_0 ph_in[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB15_0 buf15_0 buf_inv15_0 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB15_0 buf15_0 buf_inv15_0 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf15_0 buf15_0 0 100f
* buffer for ph_in[1]
XbufnA15_1 buf_inv15_1 ph_in[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
XbufpA15_1 buf_inv15_1 ph_in[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
XbufnB15_1 buf15_1 buf_inv15_1 0 0 sky130_fd_pr__nfet_01v8 w=5.2 l=0.26
XbufpB15_1 buf15_1 buf_inv15_1 vdd vdd sky130_fd_pr__pfet_01v8_hvt w=10.4 l=0.26
Cbuf15_1 buf15_1 0 100f


* Inverters for clocks
* inv for thm_sel_bld[0]
Xinvn0 clk_0_inv thm_sel_bld[0] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp0 clk_0_inv thm_sel_bld[0] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[1]
Xinvn1 clk_1_inv thm_sel_bld[1] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp1 clk_1_inv thm_sel_bld[1] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[2]
Xinvn2 clk_2_inv thm_sel_bld[2] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp2 clk_2_inv thm_sel_bld[2] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[3]
Xinvn3 clk_3_inv thm_sel_bld[3] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp3 clk_3_inv thm_sel_bld[3] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[4]
Xinvn4 clk_4_inv thm_sel_bld[4] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp4 clk_4_inv thm_sel_bld[4] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[5]
Xinvn5 clk_5_inv thm_sel_bld[5] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp5 clk_5_inv thm_sel_bld[5] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[6]
Xinvn6 clk_6_inv thm_sel_bld[6] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp6 clk_6_inv thm_sel_bld[6] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[7]
Xinvn7 clk_7_inv thm_sel_bld[7] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp7 clk_7_inv thm_sel_bld[7] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[8]
Xinvn8 clk_8_inv thm_sel_bld[8] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp8 clk_8_inv thm_sel_bld[8] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[9]
Xinvn9 clk_9_inv thm_sel_bld[9] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp9 clk_9_inv thm_sel_bld[9] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[10]
Xinvn10 clk_10_inv thm_sel_bld[10] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp10 clk_10_inv thm_sel_bld[10] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[11]
Xinvn11 clk_11_inv thm_sel_bld[11] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp11 clk_11_inv thm_sel_bld[11] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[12]
Xinvn12 clk_12_inv thm_sel_bld[12] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp12 clk_12_inv thm_sel_bld[12] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[13]
Xinvn13 clk_13_inv thm_sel_bld[13] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp13 clk_13_inv thm_sel_bld[13] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[14]
Xinvn14 clk_14_inv thm_sel_bld[14] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp14 clk_14_inv thm_sel_bld[14] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* inv for thm_sel_bld[15]
Xinvn15 clk_15_inv thm_sel_bld[15] 0 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xinvp15 clk_15_inv thm_sel_bld[15] vdd vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26


* Muxes
* mux for ph_in[0], thm_sel_bld[0]
Xmuxn0_0 buf0_0 clk_0_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp0_0 buf0_0 thm_sel_bld[0] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[0]
Xmuxn0_1 buf0_1 thm_sel_bld[0] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp0_1 buf0_1 clk_0_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[1]
Xmuxn1_0 buf1_0 clk_1_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp1_0 buf1_0 thm_sel_bld[1] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[1]
Xmuxn1_1 buf1_1 thm_sel_bld[1] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp1_1 buf1_1 clk_1_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[2]
Xmuxn2_0 buf2_0 clk_2_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp2_0 buf2_0 thm_sel_bld[2] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[2]
Xmuxn2_1 buf2_1 thm_sel_bld[2] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp2_1 buf2_1 clk_2_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[3]
Xmuxn3_0 buf3_0 clk_3_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp3_0 buf3_0 thm_sel_bld[3] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[3]
Xmuxn3_1 buf3_1 thm_sel_bld[3] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp3_1 buf3_1 clk_3_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[4]
Xmuxn4_0 buf4_0 clk_4_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp4_0 buf4_0 thm_sel_bld[4] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[4]
Xmuxn4_1 buf4_1 thm_sel_bld[4] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp4_1 buf4_1 clk_4_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[5]
Xmuxn5_0 buf5_0 clk_5_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp5_0 buf5_0 thm_sel_bld[5] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[5]
Xmuxn5_1 buf5_1 thm_sel_bld[5] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp5_1 buf5_1 clk_5_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[6]
Xmuxn6_0 buf6_0 clk_6_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp6_0 buf6_0 thm_sel_bld[6] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[6]
Xmuxn6_1 buf6_1 thm_sel_bld[6] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp6_1 buf6_1 clk_6_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[7]
Xmuxn7_0 buf7_0 clk_7_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp7_0 buf7_0 thm_sel_bld[7] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[7]
Xmuxn7_1 buf7_1 thm_sel_bld[7] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp7_1 buf7_1 clk_7_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[8]
Xmuxn8_0 buf8_0 clk_8_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp8_0 buf8_0 thm_sel_bld[8] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[8]
Xmuxn8_1 buf8_1 thm_sel_bld[8] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp8_1 buf8_1 clk_8_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[9]
Xmuxn9_0 buf9_0 clk_9_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp9_0 buf9_0 thm_sel_bld[9] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[9]
Xmuxn9_1 buf9_1 thm_sel_bld[9] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp9_1 buf9_1 clk_9_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[10]
Xmuxn10_0 buf10_0 clk_10_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp10_0 buf10_0 thm_sel_bld[10] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[10]
Xmuxn10_1 buf10_1 thm_sel_bld[10] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp10_1 buf10_1 clk_10_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[11]
Xmuxn11_0 buf11_0 clk_11_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp11_0 buf11_0 thm_sel_bld[11] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[11]
Xmuxn11_1 buf11_1 thm_sel_bld[11] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp11_1 buf11_1 clk_11_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[12]
Xmuxn12_0 buf12_0 clk_12_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp12_0 buf12_0 thm_sel_bld[12] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[12]
Xmuxn12_1 buf12_1 thm_sel_bld[12] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp12_1 buf12_1 clk_12_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[13]
Xmuxn13_0 buf13_0 clk_13_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp13_0 buf13_0 thm_sel_bld[13] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[13]
Xmuxn13_1 buf13_1 thm_sel_bld[13] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp13_1 buf13_1 clk_13_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[14]
Xmuxn14_0 buf14_0 clk_14_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp14_0 buf14_0 thm_sel_bld[14] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[14]
Xmuxn14_1 buf14_1 thm_sel_bld[14] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp14_1 buf14_1 clk_14_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26

* mux for ph_in[0], thm_sel_bld[15]
Xmuxn15_0 buf15_0 clk_15_inv ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp15_0 buf15_0 thm_sel_bld[15] ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26
* mux for ph_in[1], thm_sel_bld[15]
Xmuxn15_1 buf15_1 thm_sel_bld[15] ph_out 0 sky130_fd_pr__nfet_01v8 w=0.52 l=0.26
Xmuxp15_1 buf15_1 clk_15_inv ph_out vdd sky130_fd_pr__pfet_01v8_hvt w=1.04 l=0.26


.ends
