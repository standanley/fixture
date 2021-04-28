* Amplifier with interesting dynamic response
* From Byong's work
* https://github.com/StanfordVLSI/DaVE/blob/master/mGenero/samples/template/amplifier/personality/ctle/ctle.sp


*----------------------------------------------------------------------
* LAMBDA = 0.1 u, vdd=1.8 v
*----------------------------------------------------------------------
* Continous-time linear equalizer 
* The generation resistor is controlled by v_fz input
.SUBCKT ctle vinp vinn voutp voutn v_fz vdd vss $ctl<2> ctl<1> ctl<0>
I_1 vdd ibiasn DC 500u 
R_2 voutp vdd 1k 
R_1 voutn vdd 1k 
M_1 voutn vinp a vss nmos W='250*0.1u' L='4*0.1u' GEO=1 M=5
M_2 voutp vinn b vss nmos W='250*0.1u' L='4*0.1u' GEO=1 M=5
M_3 b ibiasn vss vss nmos W='250*0.1u' L='4*0.1u' GEO=1 M=4
M_4 a ibiasn vss vss nmos W='250*0.1u' L='4*0.1u' GEO=1 M=4
M_5 ibiasn ibiasn vss vss nmos W='250*0.1u' L='4*0.1u' GEO=1 M=4
Xnmos_var vdd b nmos_var M=8 lr='10u' wr='10u'
Xnmos_var_1 vdd a nmos_var M=8 lr='10u' wr='10u'
gvcr a b vcr pwl(1) v_fz vss 0v,1k 1.8v,3k  $ VCR

Cloadp voutp vss 0.01p
Cloadn voutn vss 0.01p
.ENDS	$ ctle1

* .subckt sw g d s
* g d s pwl(1) g 0  0,10meg  1.8,1m  level=1
* .ends

** nMOS varactor model
.SUBCKT nmos_var ng nds lr=0.2 wr=0.4
** lr and wr in [meter]
.param area='(lr*wr)*1e12'
.param pj='(lr+wr)*2*1e6'
.param Cgmin='(0.1822*pj+1.4809*area)*1e-15'
.param dCg='(-1*0.02472*pj+1.6923*area)*1e-15'
$.param dVgs=-0.161
.param dVgs=0.0
.param Vgnorm=0.538
cg ng nds 'Cgmin+dCg*(1.0+tanh((v(ng,nds)-dVgs)/Vgnorm))'
.ENDS $ nmos_var

