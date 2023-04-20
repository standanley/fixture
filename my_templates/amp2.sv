//top

{{ BEGINTESTS }}
clamping
{{ ENDTESTS }}
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

real out_noclamping = gain * in + offset;

{{ if clamping }}
assign out = todo;
{{ else }}
assign out = out_noclamping;
{{ endif }}

endmodule

