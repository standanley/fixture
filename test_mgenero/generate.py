from dave.mgenero.mgenero import ModelCreator
from dave.mgenero import mgenero

configuration_file = 'circuit.cfg'
#configuration_file = '/home/dstanley/research/DaVE/mGenero/examples/ctle/lab1/circuit.cfg'
interface_template = 'interface_template.cfg'
template = 'template.sv'
intermediate_template = 'template.intermediate.sv'
params = './params.yaml'
output = 'finished.sv'

m = ModelCreator(configuration_file, interface_template)
m.generate_model(template, intermediate_template)
m.backannotate_model(intermediate_template, output, params)

