.model EENMOS NMOS (VTO=0.4 KP=432E-6 GAMMA=0.2 PHI=.88)
.model EEPMOS PMOS (VTO=-0.4 KP=122E-6 GAMMA=0.2 PHI=.88)

.subckt sampler1 clk<0> clk<1> in_ out<0> out<1>
C1 N002 0 100p
C2 out<0> 0 10p
C4 N001 0 500p
R1 N001 clk<0> 1k
R2 N002 in_ 10k
C3 out<1> 0 10p
C5 clock1filtered 0 500p
R3 clock1filtered clk<1> 1k
M2 out<1> clock1filtered N002 0 EENMOS w=5u l=0.1u
C6 out<1> clock1filtered 500f
M1 out<0> N001 N002 0 EENMOS w=5u l=0.1u
C7 out<0> N001 500f
.ends sampler1

