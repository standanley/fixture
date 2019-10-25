/***********
* template made quickly by Daniel
* For demoing the rx with a slicer
***********/

// the declaratino comes straight from one of Byong's templates
module $$(Module.name()) #(
// parameters here
  $$(Module.parameters())
) (
  $$(Module.pins())
);

`get_timeunit
PWLMethod pm=new;

$$Pin.print_map() $$# map between user pin names and generic ones


// now parts that Daniel wrote

real slice_point = $$get_lm_equation('test1', 'slice_point');
real in_real;

pwl2real converter(
	.in(in_),
	.out(out)
);

always @(*) begin
	out <= in_real < slice_point;
end
