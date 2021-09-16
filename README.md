# fixture


[![Build Status](https://travis-ci.com/standanley/fixture.svg?branch=master)](https://travis-ci.com/standanley/fixture)


fixture is a tool to help circuit designers model mixed-signal systems. The goal is to automatically analyze analog blocks at the spice level and produce equivalent functional models in SystemVerilog. The models are pin-compatible and include the behavior of all pins, allowing the user to test complex digital-analog interaction in a fully digital simulator.

To accomplish this goal, fixture draws on a library of hand-written templates to analyze circuits and extract parameters. These parameters can then be used to create functional models using tools such as [DaVE](https://github.com/StanfordVLSI/DaVE/tree/move_to_python3) or [msdsl](https://github.com/sgherbst/msdsl).


## Templates for analysis
Recently, development on fixture has been focused on circuit analysis. The goal is to create a library of flexible analysis scripts that can cover a wide variety of circuits. The existing templates can be found in [fixture/templates](https://github.com/standanley/fixture/tree/master/fixture/templates). The main advantage of fixture is that each of these templates can cover many variations of the core circuit; for example, the differential amplifier template can analyze a basic differential amplifier as well as bias circuitry and calibration inputs, as well as CTLEs and TIAs.

The output of an analysis template is a list of extracted parameters. For a differential amplifier, these parameters might be the gain, the common-mode gain, and the pole and zero locations. These parameters can be used to verify circuit behavior, as the basis for a functional model, or to verify model behavior.

## Templates for models
Once parameters have been extracted using an analysis template, they can be inserted into a functional model. Models are typically hand-written for a specific class of circuit, with annotations for the tool to fill in extracted information. Two examples, for use with the [DaVE](https://github.com/StanfordVLSI/DaVE/tree/move_to_python3) tool mGenero, can be found in [tests/configs/amplifier.template.sv](https://github.com/standanley/fixture/blob/master/tests/configs/amplifier.template.sv) and [tests/configs/phase_blender.sv](https://github.com/standanley/fixture/blob/master/tests/configs/phase_blender.sv). fixture can automatically generate all the collateral needed for running mGenero with these templates. These two examples are written for use in a CPU-based simulation, although mGenero could also be used with models intended for FPGA emulation.

It is also possible to combine fixture with other tools for model generation. An example of integration with [msdsl](https://github.com/sgherbst/msdsl) can be found [here](https://github.com/standanley/fixture_demo_kiwi/tree/master/generated_simple_ctle_msdsl).

# Installation
fixture is written in python3, so the first step is to set up a python3 environment. fixture uses [fault](https://github.com/leonardt/fault) for interacting with simulators. Temporarily, fixture has its own development branch of fault, found [here](https://github.com/standanley/fault/tree/fixture_additions). The recommended way of installing is to use git to download the source code from the development branch, and install with pip.

    git clone https://github.com/standanley/fault/tree/fixture_additions
    cd fault
    pip install -e .
    cd ..
    
Then, install fixture the same way.

    git clone https://github.com/standanley/fixture
    cd fixture
    pip install -e .
    cd ..
   
To run fixture, the user should first create a config file with the location of necessary files, simulator information, and information about the circuit to be analyzed. Many example config files can be found in [tests/configs](https://github.com/standanley/fixture/blob/master/tests/configs). A config file can be run with

    python -m fixture.run path/to/config
