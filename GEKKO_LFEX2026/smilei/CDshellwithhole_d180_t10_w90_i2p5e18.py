import math
import numpy as np
import scipy.constants
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

##### Physical constants
eps0 = scipy.constants.epsilon_0   # Vacuum permittivity, F/m

##### Basic reference quantities
e  = scipy.constants.e             # Reference electric charge = the elementary charge [C]
me = scipy.constants.m_e           # Reference mass = the electron mass [kg]
c  = scipy.constants.c             # Reference velocity = the speed of light [m/s]
Kr = me*c**2                       # Reference energy  [(kg m^2)/s^2]
Pr = me*c                          # Reference momentum [(kg m)/s]

##### arbitrary reference quantities 
lr  =  1.053e-6                    # laser wavelength (LFEX) [m]
wr  =  2.*math.pi*c/lr             # laser angular frequency [rad/s]
Tr  =  1./wr                       # Reference time [s]
Lr  =  c/wr                        # Reference length [m]
Er  =  me*c*wr/e                   # Reference electric field [V/m]
Br  =  me*wr/e                     # Reference magnetic field [T]
Nr  =  (eps0*me*wr**2.)/(e**2.)    # Reference particle density [/m^3]
Jr  =  c*e*Nr                      # Reference current [A/m^2]

##### Variables used for unit conversions
um  = 1.e-6/Lr                     # 1 micro meter in normalized units
fs  = 1.e-15/Tr                    # 1 femto second in normalized units

##### Mesh parameters
dx = 0.048828125*um                # timestepはdx, dyで決まり、今回はtimestep=0.11・・・
dy = 0.048828125*um
nx = 8192
ny = 8192
Lx = nx * dx
Ly = ny * dy
x0 = Lx/2.
y0 = Ly/2.

shell_inside_radius = 90*um
shell_thickness = 10*um
shell_outside_radius = shell_inside_radius + shell_thickness
shell_contamination_thickness = 0.5*um
shell_hole_diameter = 90*um
shell_cut_position = x0 - math.sqrt((shell_inside_radius)**2 - (shell_hole_diameter/2.)**2)

def electron_density(x,y):
    r = math.sqrt((x - x0)**2 + (y - y0)**2)
    if (shell_inside_radius - shell_contamination_thickness < r < shell_inside_radius) and (x > shell_cut_position):
        return 10.
    elif (shell_inside_radius < r < shell_outside_radius) and (x > shell_cut_position):
        return 30.
    elif (shell_outside_radius < r < shell_outside_radius + shell_contamination_thickness) and (x > shell_cut_position):
        return 10.
    else:
        return 0.
    
def deutron_density(x,y):
    r = math.sqrt((x - x0)**2 + (y - y0)**2)
    if (shell_inside_radius < r < shell_outside_radius) and (x > shell_cut_position):
        return 30./7.
    else:
        return 0.

def carbon_density(x,y):
    r = math.sqrt((x - x0)**2 + (y - y0)**2)
    if (shell_inside_radius < r < shell_outside_radius) and (x > shell_cut_position):
        return 30./7.
    else:
        return 0.

def proton_density(x,y):
    r = math.sqrt((x - x0)**2 + (y - y0)**2)
    if (shell_inside_radius - shell_contamination_thickness < r < shell_inside_radius) and (x > shell_cut_position):
        return 10.
    elif (shell_outside_radius < r < shell_outside_radius + shell_contamination_thickness) and (x > shell_cut_position):
        return 10.
    else:
        return 0.

Main(
  geometry = "2Dcartesian",
  interpolation_order = 2,
  timestep_over_CFL = 0.99,
  simulation_time = 7000*fs,
  cell_length  = [dx, dy],
  grid_length = [ Lx, Ly ],
  number_of_patches = [32, 32],
  EM_boundary_conditions = [
      ["silver-muller","silver-muller"],
      ["silver-muller","silver-muller"],
    ],
  solve_poisson = False,
  print_every = 100,
  reference_angular_frequency_SI = wr,
)

Species(
    name = "electron",
    position_initialization = "random",
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = 100,
    mass = 1.0,
    charge = -1.0,
    number_density = electron_density,
    temperature = [7.8125e-4],
    boundary_conditions = [
       ['remove']
    ]
)

Species(
    name = "deutron",
    position_initialization = "random",
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = 100,
    mass = 1835.5*2,
    charge = 1.0,
    number_density = deutron_density,
    temperature = [7.8125e-4],
    boundary_conditions = [
       ['remove']
    ]
)

Species(
    name = "carbon",
    position_initialization = "random",
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = 100,
    mass = 1835.5*12,
    charge = 6.0,
    number_density = carbon_density,
    temperature = [7.8125e-4],
    boundary_conditions = [
       ['remove']
    ]
)

Species(
    name = "proton",
    position_initialization = "random",
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = 100,
    mass = 1835.5,
    charge = 1.0,
    number_density = proton_density,
    temperature = [7.8125e-4],
    boundary_conditions = [
       ['remove']
    ]
)

LaserGaussian2D(
    box_side        = "xmin",
    a0              = 0.7907686655992375*2,
    omega = 1,
    focus           = [Lx/2., Ly/2.],
    waist           = 30*um/math.sqrt(2*math.log(2)),
    time_envelope   = tgaussian(center=1500*fs,fwhm=1500*fs*math.sqrt(2))
)

DiagFields(
    every = 200,
    fields = ["Ex","Ey","Bz","Rho_electron","Rho_deutron","Rho_carbon","Rho_proton"]
)

DiagParticleBinning(
deposited_quantity = "weight",
every = 200,
species = ["deutron"],
axes = [
   ["x",0*um,400*um,100],
   ["y",0*um,400*um,100],
   ["ekin",0,40,1000]
]
)

Checkpoints(
    # restart_dir = "dump1",
    dump_step = 5000,
    exit_after_dump = False,
    keep_n_dumps = 1,
)