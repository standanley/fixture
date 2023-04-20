//top

{{ BEGINTESTS }}
clamping
{{ ENDTESTS }}

// the header needs to be in a specific format so the tool can parse:
// module parameters
// module name
// module io
// fixture parameters
{{ BEGINHEADER }}

module (
    parameter testparam = 5,
    parameter nodefault = required,
)
placeholder_name (
    input real in,
    output real out,
);

real gain;
real offset;

{{ if clamping }}
real vmax;
real vmin;
{{ endif }}

{{ ENDHEADER }}

// gain, offset, vmax, and vmin were all declared inside the header
// so that the tool knows they need to be set by the tool

real reg out_noclamping;
real out;

always @(*) begin
    out_noclamping = gain * in + offset;
end

{{ if clamping }}
assign out = todo;
{{ else }}
assign out = out_noclamping;
{{ endif }}

endmodule

