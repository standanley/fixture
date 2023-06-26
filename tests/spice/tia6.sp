.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

* block symbol definitions
.subckt myamp_stage inn_internal inp_internal vdd vbias outn outp
M1 vtail vbias 0 0 EENMOS l=.1u w=20u
M2 outn inp_internal vtail 0 EENMOS l=.1u w=10u
M3 outp inn_internal vtail 0 EENMOS l=.1u w=10u
R1 vdd outn 5k
R2 vdd outp 5k


* added to increase cm gain
* apparently the current source transistor M3 has no sensitivity to source-drain voltage
*R3 N001 outp 50k
*R4 N001 outn 50k
R5 vtail 0 10k

* we should probably do that on the other transistors as well
Rpnonideality outp vtail 10k
Rnnonideality outn vtail 10k


.ends myamp_stage


.subckt myinv vdd in out
Mmp out in 0 0 EENMOS l=0.1u w=1u
Mmn vdd in out vdd EEPMOS l=0.1u w=1u
.ends myinv


.subckt myamp inn inp vdd ibias outn outp rfadj<0> rfadj<1> rfadj<2> rfadj<3> rfadj<4> rfadj<5>
* just need to generate vbias, let's multiply by 10, so nominal ibias maybe 20u? no idea
Mbias ibias ibias 0 0 EENMOS l=0.1u w=2u
X1 inn inp vdd ibias outn_mid outp_mid myamp_stage
X2 outn_mid outp_mid vdd ibias outn outp myamp_stage

* trying to fix a convergence problem
Cloadmidn outn_mid 0 1p
Cloadmidp outp_mid 0 1p

* TODO inverter to create rf_adj_b
Xrfinv0 vdd rfadj<0> rfadj_b<0> myinv
Xrfinv1 vdd rfadj<1> rfadj_b<1> myinv
Xrfinv2 vdd rfadj<2> rfadj_b<2> myinv
Xrfinv3 vdd rfadj<3> rfadj_b<3> myinv
Xrfinv4 vdd rfadj<4> rfadj_b<4> myinv
Xrfinv5 vdd rfadj<5> rfadj_b<5> myinv

* n side feedback
Rrfn0 rfn0node outp 1k
Mnrfn0 inn rfadj<0> rfn0node 0   EENMOS l=0.1u w=1u
Mprfn0 inn rfadj_b<0> rfn0node vdd EEPMOS l=0.1u w=1u
Rrfn1 rfn1node outp 2k
Mnrfn1 inn rfadj<1> rfn1node 0   EENMOS l=0.1u w=1u
Mprfn1 inn rfadj_b<1> rfn1node vdd EEPMOS l=0.1u w=1u
Rrfn2 rfn2node outp 4k
Mnrfn2 inn rfadj<2> rfn2node 0   EENMOS l=0.1u w=1u
Mprfn2 inn rfadj_b<2> rfn2node vdd EEPMOS l=0.1u w=1u
Rrfn3 rfn3node outp 8k
Mnrfn3 inn rfadj<3> rfn3node 0   EENMOS l=0.1u w=1u
Mprfn3 inn rfadj_b<3> rfn3node vdd EEPMOS l=0.1u w=1u
Rrfn4 rfn4node outp 16k
Mnrfn4 inn rfadj<4> rfn4node 0   EENMOS l=0.1u w=1u
Mprfn4 inn rfadj_b<4> rfn4node vdd EEPMOS l=0.1u w=1u
Rrfn5 rfn5node outp 32k
Mnrfn5 inn rfadj<5> rfn5node 0   EENMOS l=0.1u w=1u
Mprfn5 inn rfadj_b<5> rfn5node vdd EEPMOS l=0.1u w=1u

* p side feedback
Rrfp0 rfp0node outn 1k
Mnrfp0 inp rfadj<0> rfp0node 0   EENMOS l=0.1u w=1u
Mprfp0 inp rfadj_b<0> rfp0node vdd EEPMOS l=0.1u w=1u
Rrfp1 rfp1node outn 2k
Mnrfp1 inp rfadj<1> rfp1node 0   EENMOS l=0.1u w=1u
Mprfp1 inp rfadj_b<1> rfp1node vdd EEPMOS l=0.1u w=1u
Rrfp2 rfp2node outn 4k
Mnrfp2 inp rfadj<2> rfp2node 0   EENMOS l=0.1u w=1u
Mprfp2 inp rfadj_b<2> rfp2node vdd EEPMOS l=0.1u w=1u
Rrfp3 rfp3node outn 8k
Mnrfp3 inp rfadj<3> rfp3node 0   EENMOS l=0.1u w=1u
Mprfp3 inp rfadj_b<3> rfp3node vdd EEPMOS l=0.1u w=1u
Rrfp4 rfp4node outn 16k
Mnrfp4 inp rfadj<4> rfp4node 0   EENMOS l=0.1u w=1u
Mprfp4 inp rfadj_b<4> rfp4node vdd EEPMOS l=0.1u w=1u
Rrfp5 rfp5node outn 32k
Mnrfp5 inp rfadj<5> rfp5node 0   EENMOS l=0.1u w=1u
Mprfp5 inp rfadj_b<5> rfp5node vdd EEPMOS l=0.1u w=1u

*Rrfn inn outp 10k
*Rrfp inp outn 10k
Ccfn inn outp 100p
Ccfp inp outn 100p




* random load caps
Cloadn outn 0 100p
Cloadp outp 0 100p
.ends myamp



.subckt myamp_clamped inn inp vdd ibias outn_clamped outp_clamped rfadj<0> rfadj<1> rfadj<2> rfadj<3> rfadj<4> rfadj<5>

Xorig_amp inn inp vdd ibias outn outp rfadj<0> rfadj<1> rfadj<2> rfadj<3> rfadj<4> rfadj<5> myamp

*** now just clamp outp and outn
*** center of output is around 2.28
*** we want nmos to pull up to 2.38, so we use clamp_high=2.78 
*** and pmos to pull down to 2.18, so we use clamp_low=1.78
*** I think we need to buffer these with the threshold value of 0.4
**
**Vvclamp_high clamp_high 0 2.78
**Vvclamp_low clamp_low 0 1.78
**
**Mclamp_high_n_nmos outn             clamp_high outn_clamped_mid 0     EENMOS l=0.1u w=1u
**Mclamplow_n_pmos   outn_clamped_mid clamp_low  outn_clamped     vdd   EEPMOS l=0.1u w=2u
**
**Mclamp_high_p_nmos outp             clamp_high outp_clamped_mid 0     EENMOS l=0.1u w=1u
**Mclamplow_p_pmos   outp_clamped_mid clamp_low  outp_clamped     vdd   EEPMOS l=0.1u w=2u

* we never got that clamping to work properly, so let's just use a vcvs

Ediff outdiff 0 VALUE {V(outp) - V(outn)}
Eamplitude amplitude 0 VALUE {0.1+0.08*(sin((V(vdd)-3.0)*3-0.7)+0.64)}
Ediff_clamped outdiff_clamped 0 VALUE {V(amplitude)*tanh((1/V(amplitude))*V(outdiff)) + 0.05*V(outdiff)}
Ecm outcm 0 VALUE {(V(outp)+V(outn))/2}
Epclamped outp_clamped 0 VALUE {V(outcm) + V(outdiff_clamped)/2}
Enclamped outn_clamped 0 VALUE {V(outcm) - V(outdiff_clamped)/2}


.ends myamp_clamped
