#from .. import templates
#from ..templates import *
from fixture import TemplateMaster

class SimpleAmpTemplate(TemplateMaster):
    __name__ = 'abc123'
    required_ports = ['in_single', 'out_single']

