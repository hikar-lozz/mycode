# hil4利用時はimport happiをコメントアウトして以下を実行
import os
diagnostics_path = os.path.expanduser("~/Smilei/scripts/Diagnostics.py")
exec(compile(open(diagnostics_path).read(), diagnostics_path, 'exec'))

import math
# import happi
import numpy as np
import shutil
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm
import scipy.constants

##### 1. 物理定数とインプットファイルからのパラメータ設定
e   = scipy.constants.e                # 電荷 [C]
me  = scipy.constants.m_e              # 電子質量 [kg]
eps0 = scipy.constants.epsilon_0       # 真空誘電率 [F/m]
c   = scipy.constants.c                # 光速 [m/s]

# レーザーと基準値の設定
lr  =  1.053e-6                        # レーザー波長 [m]
wr  =  2. * math.pi * c / lr           # 角周波数 [rad/s]
Tr  =  1. / wr                         # 基準時間 [s]
Lr  =  c / wr                          # 基準長さ [m]
Nr  =  (eps0 * me * wr**2.) / (e**2.)  # 基準粒子密度 [/m^3]

# インプットファイルから抽出した空間・時間パラメータ (シミュレーション側と一致)
um = 1.e-6 / Lr                     # 1 um の正規化単位値
dx = 0.048828125 * 0.25 * um        # セルサイズ x
dy = 0.048828125 * 0.25 * um        # セルサイズ y
nx = 2048                           # x方向セル数
ny = 2048                           # y方向セル数
Lx_um = nx * dx * Lr * 1e6          # x方向物理長 [um]
Ly_um = ny * dy * Lr * 1e6          # y方向物理長 [um]

# 2D Cartesian CFL条件から正規化時間刻み dt を計算
dt_cfl = 1.0 / math.sqrt(1.0 / (dx**2) + 1.0 / (dy**2))
dt = 0.99 * dt_cfl                  # timestep_over_CFL = 0.99
diag_every = 5000                   # DiagFields(every = 5000)

##### 2. フォルダの準備
data_dir = './Rho_electron_data/'
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

fig_dir = './fig_Rho_electron/'
if os.path.exists(fig_dir):
    shutil.rmtree(fig_dir)
os.makedirs(fig_dir)

##### 3. Smilei データの読み込みと一時保存
S = happi.Open('./')
timesteps = S.Field(0, 'Rho_electron').getTimesteps()

print("--- Step 1: Rho_electron データの抽出と一時保存 (.npy) ---")
for i in tqdm.tqdm(range(int(len(timesteps)))):
    Rho_raw = np.array(S.Field(0, 'Rho_electron').getData(timestep=timesteps[i]))
    
    # 配列の次元数に合わせて安全に2次元抽出
    if Rho_raw.ndim == 3:
        Rho_matrix = Rho_raw[0, :, :].T
    else:
        Rho_matrix = Rho_raw.T
        
    # 電荷密度（マイナス）の絶対値を取り、基準密度Nrをかけて物理的な電子数密度 [m^-3] へ変換
    Rho_m3 = np.abs(Rho_matrix) * Nr
    np.save(data_dir + 'Rho_electron_' + str(i) + '.npy', Rho_m3)

##### 4. 2次元カラーマップ画像の生成
print("\n--- Step 2: 2次元カラーマップ画像の生成 (.png) ---")
for i in tqdm.tqdm(range(int(len(timesteps)))):
    Rho_data = np.load(data_dir + 'Rho_electron_' + str(i) + '.npy')
    
    fig = plt.figure(figsize=(15, 12))
    plt.rcParams["font.size"] = 28
    ax1 = fig.add_subplot(111)
    
    # 物理時間 [ps] の厳密な自動計算
    actual_timestep = timesteps[i]
    time_ps = (actual_timestep * dt * Tr) * 1e12
    ax1.text(0.99, 0.99, "time = {:.2f} [ps]".format(time_ps), va='top', ha='right', transform=ax1.transAxes, color='black')
    
    # 描画（カラーマップを jet、vmin=1e25, vmax=1e30 の線形スケール）
    img1 = ax1.imshow(Rho_data, aspect='equal', origin='lower', cmap='jet', 
                      vmin=2e28, vmax=2e29, extent=[0, Lx_um, 0, Ly_um])
    
    # カラーバー設定
    cbar1 = fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"Electron Density [m$^{-3}$]")
    
    ax1.set_title('Electron Density')
    ax1.set_xlabel(r"x [$\mu$m]")
    ax1.set_ylabel(r"y [$\mu$m]")
    
    # 空間軸は extent により物理座標 [um] を直接表示
    ax1.set_xlim(0, Lx_um)
    ax1.set_ylim(0, Ly_um)

    # 保存
    fig.savefig('{}/Rho_electron_{:03d}.png'.format(fig_dir, i))
    plt.close(fig)

# 一時フォルダの削除
shutil.rmtree(data_dir)
print("\n解析が完了しました。画像は '{}' に保存されています。".format(fig_dir))