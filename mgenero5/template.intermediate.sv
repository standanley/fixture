
/****************************************************************
* This code is automatically generated by "mGenero"
* at Thu, 16 Jul 2020 12:37:23.
*
* Copyright (c) 2014-Present by Stanford University. All rights reserved.
*
* The information and source code contained herein is the property
* of Stanford University, and may not be disclosed or reproduced
* in whole or in part without explicit written authorization from
* Stanford University.
* For more information, contact bclim@stanford.edu
****************************************************************/
/********************************************************************
filename: phase_blender.sv
Description: 
multi-bit phase blender.
Assumptions:
Todo:
********************************************************************/

module myphaseblender #(
  parameter integer Nblender = 4 // # of control bits
) (
  input logic [1:0] ph_in, // ph_in
  output logic  ph_out, // ph_out
  input logic [15:0] thm_sel_bld // thm_sel_bld
);

    timeunit 1fs;
    timeprecision 1fs;

// map pins between generic names and user names, if they are different
 

// Declare parameters
real gain;
real offset;

always @(*) begin
$${
digital_modes = [get_lm_equation_modes('test1', 'gain')]
digital_cases = [digital_modes[0][0].keys()]
variable_map = {}
}$$

$$[if not mode_exists('test1')]
  gain = $$get_lm_equation('test1', 'gain');
  offset = $$get_lm_equation('test1', 'offset');
$$[else]
  case({$$(','.join(digital_cases[0]))})
$$[for m in digital_modes[0]]
  {$$(','.join(["%d'b%s" % (Pin.vectorsize(d), dec2bin('%d'%m[d], Pin.vectorsize(d))) for d in digital_cases[0]]))}: begin
    gain = $$get_lm_equation('test1', 'gain', m);
    offset = $$get_lm_equation('test1', 'offset', m);
  end
$$[end for]
  default: begin
    gain = $$get_lm_equation('test1', 'gain', digital_modes[0][0]);
    offset = $$get_lm_equation('test1', 'offset', digital_modes[0][0]);
  end
  endcase
$$[end if]

end

real wgt;
real td;
assign wgt = gain;
assign td = offset;

// fixed blender error by sjkim85 (3th May 2020) ------------------------------------------------
real rise_lead;
real rise_lag;
real fall_lead;
real fall_lag;
real rise_diff_in;
real fall_diff_in;
real ttotr;
real ttotf;
real flip;

    assign ph_and = ph_in[0]&ph_in[1];
    assign ph_or = ph_in[0]|ph_in[1];

    always @(posedge ph_in[0]) flip = ph_in[1];
    always @(negedge ph_in[0]) flip = ~ph_in[1];

    always @(posedge ph_or) begin
        rise_lead = $realtime/1s;
    end
    always @(posedge ph_and) begin
        rise_lag = $realtime/1s;
        rise_diff_in = rise_lag - rise_lead;
        ttotr = (flip+(1-2*flip)*wgt)*rise_diff_in + td - rise_diff_in;
        ph_out <= #(ttotr*1s) 1'b1;
    end
    always @(negedge ph_and) begin
        fall_lead = $realtime/1s;
    end
    always @(negedge ph_or) begin
        fall_lag = $realtime/1s;
        fall_diff_in = fall_lag - fall_lead;
        ttotf = (flip+(1-2*flip)*wgt)*fall_diff_in + td - fall_diff_in;
        ph_out <= #(ttotf*1s) 1'b0;
    end

//---------------------------------------------- ------------------------------------------------

endmodule

