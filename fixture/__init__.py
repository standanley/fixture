import collections.abc
from .real_types import RealIn, BinaryAnalog, RealOut, Array, input, output, Real # BinaryAnalogIn, TestVectorInput, TestVectorOutput
# TODO this solves a problem where magma tries to use collections.abc before
# importing it. Strangely, importing collections alone is not good enough

from .template_master import TemplateMaster
from .sampler import Sampler
from .templates import *
from .create_testbench import Testbench
from .dump_yaml import dummy_dump
from .regression import Regression
from .mgenero_interface import dump_yaml
from .run import run
