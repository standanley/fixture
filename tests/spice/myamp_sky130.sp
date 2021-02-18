.lib "../sky130/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice" tt
*.lib "/home/dstanley/research/fixture/sky130/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice" tt
*.lib "/home/dstanley/research/skywater/skywater-pdk/libraries/sky130_fd_pr/latest/models/sky130.lib.spice" tt


.subckt myamp in_ out vdd vss

R0 vdd out 1000
*MN0 out in_ vss vss EENMOS w=0.4u l=0.1u
*XN0 out in_ vss vss sky130_fd_pr__nfet_01v8 w=650000u l=150000u
XN0 out in_ vss vss sky130_fd_pr__nfet_01v8 w=10 l=.260
*XN0 out in_ vss vss sky130_fd_pr__nfet_01v8 ad=1.69e+11p pd=1.82e+06u as=1.69e+11p ps=1.82e+06u w=650000u l=150000u

.ends
