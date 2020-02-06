from dave.mgenero.mgenero import ModelCreator
from dave.mgenero import mgenero

configuration_file = 'circuit.cfg'
#configuration_file = 'circuit_from_mgenero.cfg'
#configuration_file = '/home/dstanley/research/DaVE/mGenero/examples/ctle/lab1/circuit.cfg'

interface_template = 'interface.yaml'
#interface_template = 'interface_from_mgenero.yaml'

template = 'my_template_from_mgenero.sv'
#template = 'template_from_mgenero.sv'

params = './amp_model.yaml'
#params = './amp_model_from_mgenero.yaml'

intermediate_template = 'template.intermediate.sv'
#intermediate_template = 'template.sv'
output = 'finished.sv'

m = ModelCreator(configuration_file, interface_template)
m.generate_model(template, intermediate_template)
m.backannotate_model(intermediate_template, output, params)

