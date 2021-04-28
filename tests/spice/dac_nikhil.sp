* DAC from Nikhil's EE272B project

.subckt vin vlow vref dac_8bit q0 q1 q2 q3 q4 q5 q6 q7 vcom_buf VDD GND

* from dac_8bit_commands.spice
* I removed vdd=1.8 because I'll take that from off-chip
.param Ibias_comp=10u vincm_comp=0.7 Wp_latch=8 Wp_comp=16 Wn_latch=1 Wn_mid=1
+   Lp_comp=1 Ln_latch=0.3 Ln_mid=0.3 Clsb=0.1p
+   Itail=10u Ibias=10u Wp=16 Wn=2 Wbias=1 Lbias=6.4 nfactorL=1 nfactorW=6 K=1 K2=0.5


**.subckt dac_8bit
C1 vcom cdumm 'Clsb' m=1 
C2 vcom c0m 'Clsb' m=1 
C3 vcom c1m '2*Clsb' m=1 
C4 vcom c7m '128*Clsb' m=1 
C5 vcom c3m '8*Clsb' m=1 
C6 vcom c2m '4*Clsb' m=1 
C7 vcom c4m '16*Clsb' m=1 
C8 vcom c5m '32*Clsb' m=1 
C9 vcom c6m '64*Clsb' m=1 
XM7 vcom adc_run vlow VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM8 vlow sample vcom GND sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
x33 sample adc_run inv M=1
x34 sample vin c7m net1 amux_2to1 M=1
x35 q7 vref net1 vlow amux_2to1 M=1
x36 sample vin c6m net2 amux_2to1 M=1
x37 q6 vref net2 vlow amux_2to1 M=1
x38 sample vin c5m net3 amux_2to1 M=1
x39 q5 vref net3 vlow amux_2to1 M=1
x40 sample vin c4m net4 amux_2to1 M=1
x41 q4 vref net4 vlow amux_2to1 M=1
x42 sample vin c3m net5 amux_2to1 M=1
x43 q3 vref net5 vlow amux_2to1 M=1
x44 sample vin c2m net6 amux_2to1 M=1
x45 q2 vref net6 vlow amux_2to1 M=1
x46 sample vin c1m net7 amux_2to1 M=1
x47 q1 vref net7 vlow amux_2to1 M=1
x48 sample vin c0m net8 amux_2to1 M=1
x49 q0 vref net8 vlow amux_2to1 M=1
x50 sample vin cdumm net9 amux_2to1 M=1
x51 GND vref net9 vlow amux_2to1 M=1
x1 vcom_buf vlow comp_out comp_outm adc_clk latched_comparator_folded
x2 vcom_buf vcom vcom_buf se_fold_casc_wide_swing_opamp
**** begin user architecture code

.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_01v8_lvt/sky130_fd_pr__nfet_01v8_lvt__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_03v3_nvt/sky130_fd_pr__nfet_03v3_nvt__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_05v0_nvt/sky130_fd_pr__nfet_05v0_nvt__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/esd_nfet_01v8/sky130_fd_pr__esd_nfet_01v8__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_01v8_lvt/sky130_fd_pr__pfet_01v8_lvt__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_01v8_hvt/sky130_fd_pr__pfet_01v8_hvt__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/esd_pfet_g5v0d10v5/sky130_fd_pr__esd_pfet_g5v0d10v5__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_g5v0d10v5/sky130_fd_pr__pfet_g5v0d10v5__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_g5v0d16v0/sky130_fd_pr__pfet_g5v0d16v0__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_g5v0d10v5/sky130_fd_pr__nfet_g5v0d10v5__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_g5v0d16v0/sky130_fd_pr__nfet_g5v0d16v0__tt_discrete.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/esd_nfet_g5v0d10v5/sky130_fd_pr__esd_nfet_g5v0d10v5__tt.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/models/corners/tt/nonfet.spice
* Mismatch parameters
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_01v8/sky130_fd_pr__nfet_01v8__mismatch.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_01v8/sky130_fd_pr__pfet_01v8__mismatch.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_01v8_lvt/sky130_fd_pr__nfet_01v8_lvt__mismatch.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_01v8_lvt/sky130_fd_pr__pfet_01v8_lvt__mismatch.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_01v8_hvt/sky130_fd_pr__pfet_01v8_hvt__mismatch.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_g5v0d10v5/sky130_fd_pr__nfet_g5v0d10v5__mismatch.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/pfet_g5v0d10v5/sky130_fd_pr__pfet_g5v0d10v5__mismatch.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_05v0_nvt/sky130_fd_pr__nfet_05v0_nvt__mismatch.corner.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/cells/nfet_03v3_nvt/sky130_fd_pr__nfet_03v3_nvt__mismatch.corner.spice
* Resistor~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/Capacitor
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/models/r+c/res_typical__cap_typical.spice
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/models/r+c/res_typical__cap_typical__lin.spice
* Special cells
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/models/corners/tt/specialized_cells.spice
* All models
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/models/all.spice
* Corner
.include ~/skywater-pdk/libraries/sky130_fd_pr_ngspice/latest/models/corners/tt/rf.spice

**** end user architecture code
**.ends

* expanding   symbol:
*+  /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/inv.sym # of pins=2
* sym_path: /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/inv.sym
* sch_path: /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/inv.sch
.subckt inv  A Y   M=1
*.ipin A
*.opin Y
XM2 Y A GND GND sky130_fd_pr__nfet_01v8 L=0.15 W=0.65 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=M m=M 
XM1 Y A VDD VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=M m=M 
.ends


* expanding   symbol:
*+  /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/amux_2to1.sym # of pins=4
* sym_path: /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/amux_2to1.sym
* sch_path: /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/amux_2to1.sch
.subckt amux_2to1  SEL A Y B   M=1
*.ipin A
*.ipin B
*.ipin SEL
*.opin Y
XM7 Y SELB A VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=M m=M 
XM8 A SEL Y GND sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=M m=M 
x3 SEL SELB inv M=M
XM1 Y SEL B VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=M m=M 
XM2 B SELB Y GND sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=M m=M 
.ends


* expanding   symbol:
*+  /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/latched_comparator_folded.sym # of pins=5
* sym_path:
*+ /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/latched_comparator_folded.sym
* sch_path:
*+ /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/latched_comparator_folded.sch
.subckt latched_comparator_folded  vim vip vop vom clk
*.ipin vip
*.ipin vim
*.opin vop
*.opin vom
*.ipin clk
XM5 vcompm vcompp VDD VDD sky130_fd_pr__pfet_01v8 L=1 W='Wp_latch' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM10 net3 vcompm VDD VDD sky130_fd_pr__pfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM11 net3 vcompm GND GND sky130_fd_pr__nfet_01v8 L=0.15 W=0.65 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM13 net4 vcompp GND GND sky130_fd_pr__nfet_01v8 L=0.15 W=0.65 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM15 net4 vcompp VDD VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM6 vcompp vcompm VDD VDD sky130_fd_pr__pfet_01v8 L=1 W='Wp_latch' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM12 net2 net1 GND GND sky130_fd_pr__nfet_01v8 L='Ln_latch' W='Wn_latch' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM14 net1 net2 GND GND sky130_fd_pr__nfet_01v8 L='Ln_latch' W='Wn_latch' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM3 vcompm clk VDD VDD sky130_fd_pr__pfet_01v8 L=1 W=2 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM7 vcompp clk VDD VDD sky130_fd_pr__pfet_01v8 L=1 W=2 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM8 vop vcompm_buf VDD VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM9 vop vom VDD VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM18 net5 vcompm_buf GND GND sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM19 vop vom net5 GND sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM20 vom vop VDD VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM21 vom vcompp_buf VDD VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM22 net6 vcompp_buf GND GND sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM23 vom vop net6 GND sky130_fd_pr__nfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM24 vcompm_buf net3 VDD VDD sky130_fd_pr__pfet_01v8 L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM25 vcompm_buf net3 GND GND sky130_fd_pr__nfet_01v8 L=0.15 W=0.65 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM26 vcompp_buf net4 GND GND sky130_fd_pr__nfet_01v8 L=0.15 W=0.65 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM27 vcompp_buf net4 VDD VDD sky130_fd_pr__pfet_01v8_hvt L=0.15 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
I1 net8 GND 'Ibias_comp' 
XM4 net8 net8 VDD VDD sky130_fd_pr__pfet_01v8 L=2 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM29 net7 net8 VDD VDD sky130_fd_pr__pfet_01v8 L=2 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
XM2 net1 vip net7 VDD sky130_fd_pr__pfet_01v8_lvt L=1 W='Wp_comp' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM1 net2 vim net7 VDD sky130_fd_pr__pfet_01v8_lvt L=1 W='Wp_comp' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM30 vcompm clk net1 GND sky130_fd_pr__nfet_01v8 L='Ln_mid' W='Wn_mid' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM31 vcompp clk net2 GND sky130_fd_pr__nfet_01v8 L='Ln_mid' W='Wn_mid' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult=1 m=1 
XM32 net2 clk net1 VDD sky130_fd_pr__pfet_01v8_lvt L=0.35 W=1 nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29'
+ pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W'
+ sa=0 sb=0 sd=0 mult=1 m=1 
.ends


* expanding   symbol:
*+  /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/se_fold_casc_wide_swing_opamp.sym # of pins=3
* sym_path:
*+ /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/se_fold_casc_wide_swing_opamp.sym
* sch_path:
*+ /home/users/nhpoole/ee272b/ee272b_mixed_signal_mmwave_accelerator/designs/se_fold_casc_wide_swing_opamp.sch
.subckt se_fold_casc_wide_swing_opamp  vo vip vim
*.ipin vip
*.ipin vim
*.opin vo
V6 vtail_cascn net1 0
XM7 net1 vbias4 GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM10 vcascnp vbias4 GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='2*Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM11 net11 vbias3 vcascnp GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM27 vcascnm vbias4 GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='2*Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM28 vo vbias3 vcascnm GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM36 vbias1 vbias2 net7 VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM37 net4 vbias2 net2 VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM38 net6 vbias2 net5 VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM39 vbias2 net3 vbias1 VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='3*K' m='3*K' 
XM40 net3 net3 vbias2 VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='3*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM41 net4 net4 vbias3 GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='3*Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM42 vbias3 net4 vbias4 GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='K' m='K' 
XM43 vbias4 vbias3 net9 GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM44 net10 vbias4 GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM45 net6 net6 net8 net8 sky130_fd_pr__nfet_01v8 L='Lbias' W='0.6*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='K' m='K' 
I1 net3 GND 'Ibias' 
V7 net8 GND 0
V8 net9 net10 0
XM8 vfoldp vip vtail_cascn GND sky130_fd_pr__nfet_01v8 L='K2*Lbias*nfactorL' W='Wbias*nfactorW' nf=1
+ ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)'
+ ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K'
+ 
XM12 net11 vbias2 vfoldp VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM31 vo vbias2 vfoldm VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM15 vfoldp net11 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='4*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM32 vfoldm net11 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='4*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM33 net7 vbias1 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM34 net2 vbias1 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM35 net5 vbias1 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM5 net1 vbias4 GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM9 vfoldm vim vtail_cascn GND sky130_fd_pr__nfet_01v8 L='K2*Lbias*nfactorL' W='Wbias*nfactorW' nf=1
+ ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)'
+ ps='2*int((nf+2)/2) * (W/nf + 0.29)' nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K'
+ 
XM4 vo_cs vbias4 GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='1*K' m='1*K' 
XM14 GND vo_sf vo_sf VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='4*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='18*K' m='18*K' 
XM16 vo_sf vbias1 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='0.5*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='1*K' m='1*K' 
C1 vcascnm vo_cs 0p m=1
XM6 net12 vbias1 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
V1 net12 vtail_cascp 0
XM13 net12 vbias1 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM2 vcascnm vim vtail_cascp VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM3 vcascnp vip vtail_cascp VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM17 net13 vbias4_ GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM18 vbias4_ vbias4_ GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM19 net15 net15 vbias4_ GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM20 net14 net15 net13 GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
I2 VDD net15 'Ibias' 
XM25 net14 net14 vbias2_ VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM26 vbias2_ net14 VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='4*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM21 vbias3_ net17 GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM22 net16 vbias4_ GND GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM23 net19 net15 net16 GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM24 net17 net17 vbias3_ GND sky130_fd_pr__nfet_01v8 L='Lbias*nfactorL' W='Wbias*nfactorW' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM29 net17 net19 net18 VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM30 net18 vbias1_ VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='4*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM46 net19 net19 vbias1_ VDD sky130_fd_pr__pfet_01v8 L='Lbias' W='2*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
XM47 vbias1_ vbias1_ VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='4*Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='9*K' m='9*K' 
C2 vo GND 2p m=1
XM1 vo_cs vo_cs VDD VDD sky130_fd_pr__pfet_01v8_lvt L='Lbias' W='Wbias' nf=1 ad='int((nf+1)/2) * W/nf * 0.29'
+ as='int((nf+2)/2) * W/nf * 0.29' pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)'
+ nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0 mult='3*K' m='3*K' 
.ends

.GLOBAL VDD
.GLOBAL GND
**** begin user architecture code

.include dac_8bit_commands.spice

**** end user architecture code
** flattened .save nodes
.end

