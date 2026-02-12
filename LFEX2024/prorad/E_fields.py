# -*- coding: utf-8 -*-
# 動作環境がpython2.7.18であることに注意

import numpy as np
import math

# ============================================================
# 定数定義
# ============================================================
N_A = 6.022e23                      # アボガドロ定数 [1/mol]
epsilon0 = 8.854e-14                # 真空の誘電率 [F/cm]
k = 1.0/(4.0*np.pi*epsilon0)        # クーロン定数
SI_to_StatVcm = 3.336e-3            # V/cm → StatV/cm換算
k_SI_to_StatVcm = k * SI_to_StatVcm
eps = 1e-10                         # しきい値 [cm^2] : 100 nm

# ---- タンタル (Ta) ----
Z_Ta = 73.0
rho_Ta = 16.65                      # g/cm^3
A_Ta = 180.95                       # g/mol
N_Ta = rho_Ta * N_A / A_Ta          # atoms/cm^3

# ---- 鉄 (Fe) ----
Z_Fe = 26.0
rho_Fe = 7.87
A_Fe = 55.845
N_Fe = rho_Fe * N_A / A_Fe

# ============================================================
# ジオメトリ定義
# ============================================================
# ---- タンタルシールド ----
shield_half_size = 0.05                             # 0.5 mm = 0.05 cm
shield_zmin = 0.050                                 # 0.5 mm
shield_zmax = 0.055                                 # 0.55 mm
shield_zcenter = 0.5 * (shield_zmin + shield_zmax)  
hole_radius = 0.01                                  # 100 µm

# ---- 鉄ナイフ ----
knife_xmin, knife_xmax = -0.2, 0.2
bias = 0.0                                          # 0.0 or -0.02
knife_ymin, knife_ymax = -0.2+bias, 0.0+bias
knife_zmin, knife_zmax = 0.281, 0.319               # 2.81 - 3.19 mm
knife_zcenter = 0.5 * (knife_zmin + knife_zmax) 

# ---- ナイフ用平面群 ----
planes = [
    (0.0, 0.0190, -0.217, 0.0638 + 0.0190*(-bias)),
    (0.0, 0.0190,  0.217, -0.0664 + 0.0190*(-bias)),
    (0.0, 1.000,  -3.291, 0.984 + 1.000*(-bias)),
    (0.0, 1.000,   3.291, -0.991 + 1.000*(-bias))
]

# ============================================================
# 点電荷設定
# ============================================================
# ---- 総電荷量 ----
Q_total_shield = 1e-12              # 1 pC
Q_total_knife  = 1e-12              # 1 pC

# ---- 配置間隔 ----
charge_spacing = 0.003              # 30 µm

'''
# ---- シールド点電荷 ----
nx_shield = int((2*shield_half_size) / charge_spacing) + 1
ny_shield = int((2*shield_half_size) / charge_spacing) + 1
valid_points_shield = []
for i in range(nx_shield):
    for j in range(ny_shield):
        xq_shield = -shield_half_size + i*charge_spacing
        yq_shield = -shield_half_size + j*charge_spacing
        if math.sqrt(xq_shield**2 + yq_shield**2) >= hole_radius:
            valid_points_shield.append((xq_shield, yq_shield)) 
sh_arr = np.array(valid_points_shield, dtype=np.float64)
    
n_shield = len(valid_points_shield)
if n_shield == 0:
    q_per_charge_shield = 0.0
else:
    q_per_charge_shield = Q_total_shield / float(n_shield)
A = k_SI_to_StatVcm * q_per_charge_shield
'''

# ---- ナイフ点電荷 ----
limit_field = 0.0                   # 0.0 or 0.10
nx_knife = int((knife_xmax - knife_xmin)/charge_spacing) + 1
ny_knife = int((knife_ymax - (knife_ymin + limit_field))/charge_spacing) + 1
valid_points_knife = []
for i in range(nx_knife):
    for j in range(ny_knife):
        xq_knife = knife_xmin + i*charge_spacing
        yq_knife = (knife_ymin + limit_field) + j*charge_spacing
        valid_points_knife.append((xq_knife, yq_knife))
kn_arr = np.array(valid_points_knife, dtype=np.float64)

q_per_charge_knife = Q_total_knife / (nx_knife * ny_knife)
B = k_SI_to_StatVcm * q_per_charge_knife

# ============================================================
# フィールド定義関数
# ============================================================
def fields(coord):
    x, y, z = coord
    Ex = Ey = Ez = 0.0
    Bx = By = Bz = 0.0
    nu = 0.0; Z = 0.0; N = 0.0

    _sh_half = shield_half_size
    _sh_zmin = shield_zmin
    _sh_zmax = shield_zmax
    _sh_hole = hole_radius
    _Z_Ta = Z_Ta; _N_Ta = N_Ta

    _kn_xmin = knife_xmin; _kn_xmax = knife_xmax
    _kn_ymin = knife_ymin; _kn_ymax = knife_ymax
    _kn_zmin = knife_zmin; _kn_zmax = knife_zmax
    _planes = planes
    _Z_Fe = Z_Fe; _N_Fe = N_Fe

    _eps = eps
    '''
    _sh_arr = sh_arr
    _sz = shield_zcenter
    _A = A
    '''
    _kn_arr = kn_arr
    _kz = knife_zcenter
    _B = B

    _np_sqrt = np.sqrt
    _np_sum = np.sum

    # ---- タンタルシールド判定 ----
    inside_xy = (abs(x) <= _sh_half) and (abs(y) <= _sh_half)
    inside_z = (_sh_zmin <= z <= _sh_zmax)
    in_hole = (math.hypot(x, y) < _sh_hole)
    if inside_xy and inside_z and not in_hole:
        Z = _Z_Ta
        N = _N_Ta

    # ---- 鉄ナイフ判定 ----
    inside_box = ((_kn_xmin <= x <= _kn_xmax) and
                  (_kn_ymin <= y <= _kn_ymax) and
                  (_kn_zmin <= z <= _kn_zmax))
    if inside_box:
        inside_planes = all((a*x + b*y + c*z + d) < 0.0 for a, b, c, d in _planes)
        if inside_planes:
            Z = _Z_Fe
            N = _N_Fe
    
    # ---- シールドおよびナイフ中心面の電場計算 ----
    if (abs(x) <= 0.17) and (abs(y) <= 0.17):
        '''
        if _sh_arr.size:
            xq_shield = _sh_arr[:, 0]
            yq_shield = _sh_arr[:, 1]
            dx = x - xq_shield
            dy = y - yq_shield
            dz = z - _sz
            r2 = dx**2 + dy**2 + dz**2
            mask = r2 > _eps
            if mask.any():
                dxm = dx[mask]
                dym = dy[mask]
                r2m = r2[mask]
                r3m = r2m * _np_sqrt(r2m)
                _C = _A / r3m
                Ex += float(_np_sum(_C * dxm))
                Ey += float(_np_sum(_C * dym))
                Ez += float(_np_sum(_C) * dz)
        '''
        
        if _kn_arr.size:
            xq_knife = _kn_arr[:, 0]
            yq_knife = _kn_arr[:, 1]
            dx = x - xq_knife
            dy = y - yq_knife
            dz = z - _kz
            r2 = dx**2 + dy**2 + dz**2
            mask = r2 > _eps
            if mask.any():
                dxm = dx[mask]
                dym = dy[mask]
                r2m = r2[mask]
                r3m = r2m * _np_sqrt(r2m)
                _D = _B / r3m
                Ex += float(_np_sum(_D * dxm))
                Ey += float(_np_sum(_D * dym))
                Ez += float(_np_sum(_D) * dz)

    return (Ex, Ey, Ez, Bx, By, Bz, nu, Z, N)