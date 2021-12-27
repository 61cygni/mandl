# --
# File: globalconfig.py
# Date: Sun Dec 26 10:33:54 PST 2021 
#
# Config parameters everyone needs to know about
#
#
# --

from decimal import *

# defaults if not specificed by user
GLOBAL_PRECISION = 500
GLOBAL_MAXITER   = (2 << 10) 

hpf = Decimal
getcontext().prec = GLOBAL_PRECISION
