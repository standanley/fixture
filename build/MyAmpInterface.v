module corebit_not (input in, output out);
  assign out = ~in;
endmodule

module MyAmpInterface (input my_in, output my_out);
wire not_inst0_out;
corebit_not not_inst0(.in(my_in), .out(not_inst0_out));
assign my_out = not_inst0_out;
endmodule

