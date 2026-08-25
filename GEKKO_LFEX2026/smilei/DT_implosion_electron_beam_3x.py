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

# rの最小・最大（正規化済）
r_min = min(R_m_normalized)
r_max = max(R_m_normalized)

# 最小点での密度を取得（固定値とする）
DT_min_val = rho[0] / 1000 * 1e6 / 1837.4 / 5 / me / Nr
E_min_val = rho[0] * 2 / 1000 * 1e6 / 1837.4 / 5 / me / Nr

# PCHIP補完ver
pchip_DT = PchipInterpolator(R_m_normalized, DT_density_normalized, extrapolate=True)
pchip_E  = PchipInterpolator(R_m_normalized, E_density_normalized, extrapolate=True)

# ラップ：r < r_min では定数を返す
def rho_DT(r):
    r = np.asarray(r)
    raw = np.where(r < r_min, DT_min_val, pchip_DT(r))
    return np.minimum(raw, 100)

def rho_E(r):
    r = np.asarray(r)
    raw = np.where(r < r_min, E_min_val, pchip_E(r))
    return np.minimum(raw, 200)

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
  print_every = 100,
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
    #particles_per_cell = electron_ppc,
    particles_per_cell = 800,
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
    #particles_per_cell = deuteron_ppc,
    particles_per_cell = 80,
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
    #particles_per_cell = tritium_ppc,
    particles_per_cell = 80,
    mass = 1837.4*3,
    charge = 1.0,
    number_density = tritium_density,
    temperature = [7.8125e-4],
    boundary_conditions = [
       ['remove']
    ]
)

# ============================================================
# ParticleInjector に必要な（依存する）namelist部品一式
#  - ここでは「電子を xmin から注入」
#  - 運動量分布：maxwell-juettner（相対論的 Maxwell）
#  - x方向にドリフト：E_drift = 4.91 MeV を仮定（= バルク運動エネルギー）
#  - （もし「温度」も 4.91 MeV にしたいなら theta_T も 9.61 を使う）
# ============================================================

# ---- 物理定数（SI） ----
c = 299792458.0  # [m/s]

# ---- あなたの系の規格化（外部で定義済みならここは不要）----
# Tr: 基準時間 [s]
# Lr: 基準長さ [m]
# Nr: 基準密度 [m^-3]
# Ly: y方向長さ（規格化 or SI のどちらか、あなたの式に合わせる）
# ※すでに上流で定義しているなら、この4行は消してください。
# Tr = ...
# Lr = ...
# Nr = ...
# Ly = ...

# ============================================================
# 1) パルス時間エンベロープ（SI→規格化）
# ============================================================
tau_si = 1.0e-12  # [s] FWHM
t0_si  = 1.0e-12  # [s] 中心
tau = tau_si / Tr
t0  = t0_si  / Tr

# ============================================================
# 2) ドリフト速度 beta を「ドリフト運動エネルギー」から作る
#    E_drift = 4.91 MeV を x方向の平行運動（バルク）として解釈
# ============================================================
E_drift_MeV = 4.91
me_c2_MeV   = 0.511
gamma_drift = 1.0 + E_drift_MeV / me_c2_MeV
beta_n_0    = np.sqrt(1.0 - 1.0 / (gamma_drift**2))  # ~0.995547...

# ============================================================
# 3) （オプション）熱温度を 4.91 MeV にする場合の規格化温度 theta
#    theta = kT / (m_e c^2) = 4.91/0.511 ~ 9.61
#    ※「温度4.91MeVの電子」という表現が“熱”を指すならこちらを使う
# ============================================================
theta_T = E_drift_MeV / me_c2_MeV  # 9.61...

# ============================================================
# 4) 種（Species）定義：Injector が species を参照するので必要
#    すでに electron species を別で定義しているなら、この Species は不要
# ============================================================
"""
Species(
    name = "electron",
    position_initialization = "random",
    momentum_initialization = "maxwell-juettner",  # injector側で上書きしてもOK
    mean_velocity = [0., 0., 0.],                  # injector側で上書き
    temperature = [1.0e-3],                        # injector側で上書き
    particles_per_cell = 1,                        # injector側で上書き
    mass = 1.0,
    charge = -1.0,
)
"""
# ============================================================
# 5) 注入数（実粒子数）から number_density を作る
#    N_target_0 = 7.627...e12（今回）
# ============================================================
N_target_0 = 7.627098665328835e12 * 3

# あなたの式：N = n * Nr * (Ly*Lr) * (beta*c) * (tau_si * sqrt(pi/(4 ln2)))
# ここで (Ly*Lr) は「境界面積（2Dなら長さ）」のつもりの項
number_density_scale = N_target_0 / (
    Nr * (Ly * Lr) * (beta_n_0 * c) * (tau_si * np.sqrt(np.pi/(4.0*np.log(2.0))))
)

# ============================================================
# 6) Particle Injector 本体（必要部分）
# ============================================================
ParticleInjector(
    name = "inj_0",
    species = "electron",
    box_side = "xmin",
    time_envelope = tgaussian(center=t0, fwhm=tau),

    # 相対論的 Maxwell（Jüttner）
    momentum_initialization = "maxwell-juettner",

    # x方向ドリフト（v/c）
    mean_velocity = [beta_n_0, 0., 0.],

    # 温度：
    # - 「熱温度 4.91 MeV」なら theta_T（=9.61）を入れる
    # - 「熱は不要で単色ビーム」なら極小（例: 1e-30）
    temperature = [theta_T],

    number_density = number_density_scale,
    particles_per_cell = 1,
)

DiagFields(
    every = 200,
    fields = ["Ex","Ey","Bz","Jx_electron","Jy_electron","Rho_electron","Rho_deuteron","Rho_tritium"]
)
