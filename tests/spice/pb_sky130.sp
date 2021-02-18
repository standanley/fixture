.lib "../sky130/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice" tt

.subckt phase_blender ph_in[1] ph_in[0] thm_sel_bld[0] thm_sel_bld[1] thm_sel_bld[2] thm_sel_bld[3] ph_out vdd



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


.ends
