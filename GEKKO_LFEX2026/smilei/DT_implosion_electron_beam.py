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
dx = 0.048828125*0.25*um
dy = 0.048828125*0.25*um
nx = 2048
ny = 2048
Lx = nx * dx
Ly = ny * dy

#初期プラズマ密度

R_um = np.array([
    1.131, 2.796, 3.557, 4.288, 4.951, 11.416, 13.259, 14.411, 15.242,
    15.922, 16.511, 17.036, 17.517, 17.963, 18.382, 18.777, 19.155,
    19.519, 19.872, 20.215
])
rho = np.array([
    0.083, 0.192, 0.316, 0.323, 0.342, 11.736, 18.126, 23.08, 27.884,
    30.874, 32.917, 34.49, 35.601, 36.366, 36.97, 37.42, 37.646,
    37.648, 37.465, 37.149
])

DT_density_normalized = rho / 1000 * 1000000 / 1837.4 / 5 / me / Nr
E_density_normalized = rho * 2 / 1000 * 1000000 / 1837.4 / 5 / me / Nr
R_m_normalized = R_um * 1e-6 / Lr

# rの最小・最大
r_min = min(R_m_normalized)
r_max = max(R_m_normalized)

# 最小点での密度を取得（固定値とする）
DT_min_val = rho[0] / 1000 * 1e6 / 1837.4 / 5 / me / Nr
E_min_val = rho[0] * 2 / 1000 * 1e6 / 1837.4 / 5 / me / Nr

# PCHIP補間ver（離散的な密度データをPCHIP補間によって連続的なデータに変換）
pchip_DT = PchipInterpolator(R_m_normalized, DT_density_normalized, extrapolate=True)
pchip_E  = PchipInterpolator(R_m_normalized, E_density_normalized, extrapolate=True)

# 密度分布関数（r < r_min では定数を返す, 密度の上限はスキン長を短くし過ぎないように設定している）
def rho_DT(r):
    r = np.asarray(r)
    raw = np.where(r < r_min, DT_min_val, pchip_DT(r))
    return np.minimum(raw, 94)

def rho_E(r):
    r = np.asarray(r)
    raw = np.where(r < r_min, E_min_val, pchip_E(r))
    return np.minimum(raw, 188)

# 密度分布関数（中心座標固定）
def electron_density(x, y):
    r = np.sqrt((x - 12.5 * um)**2 + (y - 12.5 * um)**2)
    return float(rho_E(r))

def deuteron_density(x, y):
    r = np.sqrt((x - 12.5 * um)**2 + (y - 12.5 * um)**2)
    return float(rho_DT(r))

def tritium_density(x, y):
    r = np.sqrt((x - 12.5 * um)**2 + (y - 12.5 * um)**2)
    return float(rho_DT(r)) 

Main(
  geometry = "2Dcartesian",
  interpolation_order = 2,
  timestep_over_CFL = 0.99,
  simulation_time = 5000*fs,
  cell_length  = [dx, dy],
  grid_length = [Lx, Ly],
  number_of_patches = [32, 32],
  EM_boundary_conditions = [
      ["silver-muller","silver-muller"],
      ["silver-muller","silver-muller"],
    ],
  solve_poisson = False,
  print_every = 1000,
  reference_angular_frequency_SI = wr,
)

LoadBalancing(
    initial_balance = False,
    every = 500,
)

Species(
    name = "electron",
    position_initialization = "random",
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = 80, # 800→80
    mass = 1.0,
    charge = -1.0,
    number_density = electron_density,
    temperature = [7.8125e-4],
    boundary_conditions = [
       ['remove']
    ]
)

Species(
    name = "deuteron",
    position_initialization = "random",
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = 10, # 80→10
    mass = 1837.4*2,
    charge = 1.0,
    number_density = deuteron_density,
    temperature = [7.8125e-4],
    boundary_conditions = [
       ['remove']
    ]
)

Species(
    name = "tritium",
    position_initialization = "random",
    momentum_initialization = "maxwell-juettner",
    particles_per_cell = 10, # 80→10
    mass = 1837.4*3,
    charge = 1.0,
    number_density = tritium_density,
    temperature = [7.8125e-4],
    boundary_conditions = [
       ['remove']
    ]
)

# ============================================================
# 1) 電子ビームの時間プロファイル（step1の計算に合わせ、t=1 psをピークとして時間幅1 psで起きるようにする）
# ============================================================
tau_si = 1.0e-12  # [s] FWHM
t0_si  = 1.0e-12  # [s] 中心
tau = tau_si / Tr
t0  = t0_si  / Tr

# ============================================================
# 2) 電子温度を 4.91 MeV にする場合の規格化温度 theta
#    step1においてx = 1250 umを通過する電子のエネルギースペクトルから得た温度
#    theta = T_e / (m_e c^2) = 4.91/0.511 ~ 9.61
# ============================================================
T_e = 4.91
me_c2_MeV = 0.511
theta_T = T_e / me_c2_MeV  # 9.61...

# ============================================================
# 3) 電子のx方向の平均速度（0でいいらしい）
# ============================================================
# v_x_ave = ? (m/s)
# beta_n_0 = v_x_ave / c

# ============================================================
# 4) 電子注入数を7.627098665328835e12にする場合の電子密度（電子1個当たりの平均エネルギーが4.91 MeVであると仮定）
#    本来であれば、電子のエネルギースペクトルから直接得た平均エネルギーを用いるべきだが、ここでは暫定的に4.91 MeVを用いる
#    N_e_per_dt = A*v_e*dt*n_e(t)
#    N_e = A*v_e*∫n_e(t)dt (-inf~infまで足しあわせる)
#        = A*v_e*n_e_0*τFWHM*sqrt(π/4ln2)（ガウシアンの場合）
# ============================================================
A = np.pi * ((Ly * Lr) / 2.0)**2
v_e = c
N_e = 7.627098665328835e12 * 2
n_e_0 = N_e / (A * v_e * tau_si * np.sqrt(np.pi/(4.0*np.log(2.0))))
number_density_scale = n_e_0 / Nr

ParticleInjector(
    name = "inj_0",
    species = "electron",
    box_side = "xmin",
    time_envelope = tgaussian(center=t0, fwhm=tau),
    momentum_initialization = "maxwell-juettner",
    mean_velocity = [0., 0., 0.],
    temperature = [theta_T, 1e-6, 1e-6],
    number_density = number_density_scale,
    particles_per_cell = 10,
)

DiagFields(
    every = 5000,
    fields = ["Ex","Ey","Bz","Jx_electron","Jy_electron","Rho_electron","Rho_deuteron","Rho_tritium"]
)

DiagParticleBinning(
    deposited_quantity = "weight",
    every = 5000,
    species = ["electron"],
    axes = [
        ["px", 0, 50, 200] 
    ]
)

DiagParticleBinning(
    deposited_quantity = "weight",
    every = 5000,
    species = ["electron"],
    axes = [
        ["py", -25, 25, 200]
    ]
)

Checkpoints(
    # restart_dir = "dump1",
    dump_step = 8750,
    exit_after_dump = False,
    keep_n_dumps = 1,
)