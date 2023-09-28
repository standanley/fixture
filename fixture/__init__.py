import collections.abc
# TODO this solves a problem where magma tries to use collections.abc before
# importing it. Strangely, importing collections alone is not good enough

from .representation import Representation
from .temp_channel_data_generator import ChannelUtil
from .tester import Tester
from .regression import Regression
from .template_master import TemplateMaster
from .plot_helper import PlotHelper
from .sampler import Sampler
#from .templates import *
from .create_testbench import Testbench
from .dump_yaml import dummy_dump
from .mgenero_interface import dump_yaml
from .run import run
