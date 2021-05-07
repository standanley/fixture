* Amplifier with interesting dynamic response
* From Byong's work
* https://github.com/StanfordVLSI/DaVE/blob/master/mGenero/samples/template/amplifier/personality/ctle/ctle.sp


* Not sure about these MOSFET models, especially capacitance
.model NMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88, CGSO=5e-10, CGDO=5e-10)
.model PMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88, CGSO=5e-10, CGDO=5e-10)

*----------------------------------------------------------------------
* LAMBDA = 0.1 u, vdd=1.8 v
*----------------------------------------------------------------------
* Continous-time linear equalizer 
* The generation resistor is controlled by v_fz input
.nodeset v(voutp) 0.9
.nodeset v(voutn) 0.9
.SUBCKT ctle vinp vinn voutp voutn v_fz vdd vss 
* $ctl<2> ctl<1> ctl<0>
I_1 vdd ibiasn DC 500u 
R_2 voutp vdd 1k 
R_1 voutn vdd 1k 
* NOTE I removed GEO=1 from the original netlist for the next 5 MOS
* GEO seems to be hspice-specific
* I also removed M=5 or M=4 and moved it to the width
*M_1 voutn vinp a vss nmos W={250*0.1u*5} L={4*0.1u}
M_1 voutn vinp a vss nmos W=125u L=.4u
M_2 voutp vinn b vss nmos W={250*0.1u*5} L={4*0.1u}
M_3 b ibiasn vss vss nmos W={250*0.1u*4} L={4*0.1u}
M_4 a ibiasn vss vss nmos W={250*0.1u*4} L={4*0.1u}
M_5 ibiasn ibiasn vss vss nmos W={250*0.1u*4} L={4*0.1u}
Xnmos_var vdd b nmos_var M=8 lr={10u} wr={10u}
Xnmos_var_1 vdd a nmos_var M=8 lr={10u} wr={10u}

* This is hspice syntax for a pwl voltage-controlled-resistor with 
* a PWL-defined resistance
*gvcr a b vcr pwl(1) v_fz vss 0v,1k 1.8v,3k  $ VCR
* Here is my ngspice-friendly replacement EDIT bad things happen when R==0
*gvcr a b cur='v(a,b)/(v(v_fz,vss)/1.8*3k)'
* attempt 2
evcr a temp_resistor vol='i(Rtemp) * (v(v_fz,vss)/1.8*3k)'
Rtemp temp_resistor b 100
* this capacitor is necessary for convergence in ngspice
Ctemp a b 100f
*Ctemp2 vss vdd 100p
*Rtemp a b 1.5k

Cloadp voutp vss 0.01p
Cloadn voutn vss 0.01p


.ENDS	
* $ ctle1

* .subckt sw g d s
* g d s pwl(1) g 0  0,10meg  1.8,1m  level=1
* .ends

** nMOS varactor model
.SUBCKT nmos_var ng nds lr=0.2 wr=0.4
** lr and wr in [meter]
.param area={(lr*wr)*1e12}
.param pj={(lr+wr)*2*1e6}
.param Cgmin={(0.1822*pj+1.4809*area)*1e-15}
.param dCg={(-1*0.02472*pj+1.6923*area)*1e-15}
* $.param dVgs=-0.161
.param dVgs=0.0
.param Vgnorm=0.538
cg ng nds {Cgmin+dCg*(1.0+tanh((v(ng,nds)-dVgs)/Vgnorm))}
.ENDS 
* $ nmos_var

