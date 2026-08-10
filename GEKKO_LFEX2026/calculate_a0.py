import math
import numpy as np
import scipy.constants
# With LFEX LASER , at a_0 = 1 I_0 = 1.255e^18 [W/cm^2] E_0 = 3.062e^12 [V/m] = 3.062 [TV/m]

##### Physical constants

lambda0             = 1.053e-6                  # reference length, m
c                   = scipy.constants.c         # lightspeed, m/s
omega0              = 2*math.pi*c/lambda0       # laser angular frequency, rad/s
eps0                = scipy.constants.epsilon_0 # Vacuum permittivity, F/m
e                   = scipy.constants.e         # Elementary charge, C
me                  = scipy.constants.m_e       # Electron mass, kg
ncrit               = eps0*omega0**2*me/e**2    # reference density, m-3
c_over_omega0       = lambda0/2./math.pi        # converts from c/omega0 units to m
reference_frequency = omega0                    # reference frequency, s-1
E0                  = me*omega0*c/e             # reference electric field, V/m

##### Variables used for unit conversions
c_normalized        = 1.                        # speed of light in vacuum in normalized units
um                  = 1.e-6/c_over_omega0       # 1 micron in normalized units
fs                  = 1.e-15*omega0             # 1 femtosecond in normalized units
mm_mrad             = um                        # 1 millimeter-milliradians in normalized units
pC                  = 1.e-12/e                  # 1 picoCoulomb in normalized units

I = 7.715806912141135e+17                                  # レーザー強度

a0 = e/me/c/omega0*math.sqrt(2*10**(4)*I/c/eps0)       # 無次元量a0(SI単位系）

print(a0)