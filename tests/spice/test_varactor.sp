// varactor test circuit
simulator lang=spectre
ahdl_include "ctle_varactor.va"
Vt (n 0) vsource mag=1
Cv (n 0) varactor c0=1pF c1=0.5pF v0=0 v1=1
capacitanceInF ac freq=1/(2*M_PI) start=2 stop=2 dev=Vt param=dc
save Cv:1

