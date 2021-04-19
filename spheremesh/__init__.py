from .fit import *
from .flow import *

try:
    from .plot import *
    
    has_plot = True
except ImportError:
    has_plot = False


