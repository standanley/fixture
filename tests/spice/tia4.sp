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


* RESISTORS
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




* CAPACITORS
* n side feedback
Llfn0 cfn0nodeL cfn0node 100n
Ccfn0 cfn0nodeL outp 1p
Mncfn0 inn cfadj<0> cfn0node 0   EENMOS l=0.1u w=1u
Mpcfn0 inn cfadj_b<0> cfn0node vdd EEPMOS l=0.1u w=1u
Llfn1 cfn1nodeL cfn1node 100n
Ccfn1 cfn1nodeL outp 2p
Mncfn1 inn cfadj<1> cfn1node 0   EENMOS l=0.1u w=1u
Mpcfn1 inn cfadj_b<1> cfn1node vdd EEPMOS l=0.1u w=1u
Llfn2 cfn2nodeL cfn2node 100n
Ccfn2 cfn2nodeL outp 4p
Mncfn2 inn cfadj<2> cfn2node 0   EENMOS l=0.1u w=1u
Mpcfn2 inn cfadj_b<2> cfn2node vdd EEPMOS l=0.1u w=1u
Llfn3 cfn3nodeL cfn3node 100n
Ccfn3 cfn3nodeL outp 8p
Mncfn3 inn cfadj<3> cfn3node 0   EENMOS l=0.1u w=1u
Mpcfn3 inn cfadj_b<3> cfn3node vdd EEPMOS l=0.1u w=1u
Llfn4 cfn4nodeL cfn4node 100n
Ccfn4 cfn4nodeL outp 8p
Mncfn4 inn cfadj<4> cfn4node 0   EENMOS l=0.1u w=1u
Mpcfn4 inn cfadj_b<4> cfn4node vdd EEPMOS l=0.1u w=1u
Llfn5 cfn5nodeL cfn5node 100n
Ccfn5 cfn5nodeL outp 8p
Mncfn5 inn cfadj<5> cfn5node 0   EENMOS l=0.1u w=1u
Mpcfn5 inn cfadj_b<5> cfn5node vdd EEPMOS l=0.1u w=1u


* p side feedback
Llfp0 cfp0nodeL cfp0node 100n
Ccfp0 cfp0nodeL outn 1p
Mncfp0 inp cfadj<0> cfp0node 0   EENMOS l=0.1u w=1u
Mpcfp0 inp cfadj_b<0> cfp0node vdd EEPMOS l=0.1u w=1u
Llfp1 cfp1nodeL cfp1node 100n
Ccfp1 cfp1nodeL outn 2p
Mncfp1 inp cfadj<1> cfp1node 0   EENMOS l=0.1u w=1u
Mpcfp1 inp cfadj_b<1> cfp1node vdd EEPMOS l=0.1u w=1u
Llfp2 cfp2nodeL cfp2node 100n
Ccfp2 cfp2nodeL outn 4p
Mncfp2 inp cfadj<2> cfp2node 0   EENMOS l=0.1u w=1u
Mpcfp2 inp cfadj_b<2> cfp2node vdd EEPMOS l=0.1u w=1u
Llfp3 cfp3nodeL cfp3node 100n
Ccfp3 cfp3nodeL outn 8p
Mncfp3 inp cfadj<3> cfp3node 0   EENMOS l=0.1u w=1u
Mpcfp3 inp cfadj_b<3> cfp3node vdd EEPMOS l=0.1u w=1u
Llfp4 cfp4nodeL cfp4node 100n
Ccfp4 cfp4nodeL outn 16p
Mncfp4 inp cfadj<4> cfp4node 0   EENMOS l=0.1u w=1u
Mpcfp4 inp cfadj_b<4> cfp4node vdd EEPMOS l=0.1u w=1u
Llfp5 cfp5nodeL cfp5node 100n
Ccfp5 cfp5nodeL outn 32p
Mncfp5 inp cfadj<5> cfp5node 0   EENMOS l=0.1u w=1u
Mpcfp5 inp cfadj_b<5> cfp5node vdd EEPMOS l=0.1u w=1u






*Rrfn inn outp 10k
*Rrfp inp outn 10k
*Ccfn inn outp 100p
*Ccfp inp outn 100p




* random load caps
Cloadn outn 0 100p
Cloadp outp 0 100p
.ends myamp
