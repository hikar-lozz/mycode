# hil5利用時はimport happiをコメントアウトして以下を実行
diagnostics_path = "/media/ymraid5-5/Kensei/Smilei/scripts/Diagnostics.py"
exec(compile(open(diagnostics_path).read(), diagnostics_path, 'exec'))

import math
# import happi
import numpy as np
import shutil
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import tqdm
import scipy.constants

##### 1. 物理定数とインプットファイルからのパラメータ設定
e   = scipy.constants.e               # 電荷 [C]
me  = scipy.constants.m_e             # 電子質量 [kg]
c   = scipy.constants.c               # 光速 [m/s]

# レーザーと基準値の設定
lr  =  1.053e-6                       # レーザー波長 [m]
wr  =  2. * math.pi * c / lr         # 角周波数 [rad/s]
Tr  =  1. / wr                       # 基準時間 [s]
Lr  =  c / wr                        # 基準長さ [m]
Er  =  me * c * wr / e               # 基準電場 [V/m] (Bzの c 倍)

# インプットファイルから抽出した空間・時間パラメータ (シミュレーション側と一致)
um = 1.e-6 / Lr                      # 1 um の正規化単位値
dx = 0.048828125 * um                 # セルサイズ x
dy = 0.048828125 * um                 # セルサイズ y
nx = 4096                            # x方向セル数
ny = 4096                            # y方向セル数

# 2D Cartesian CFL条件から正規化時間刻み dt を計算
dt_cfl = 1.0 / math.sqrt(1.0 / (dx**2) + 1.0 / (dy**2))
dt = 0.99 * dt_cfl                   # timestep_over_CFL = 0.99
diag_every = 2000                    # DiagFields(every = 2000)

##### Smilei データの読み込み
S = happi.Open('./')

# 描画するフィールドのリスト
fields = ['Ex', 'Ey']

# ★カラーバーの範囲を固定 [-1 〜 1]
vmin_val = -0.4
vmax_val = 0.4

for field in fields:
    print(f"\n========== {field} の解析を開始 ==========")
    
    ##### 2. フォルダの準備
    data_dir = f'./{field}_data/'
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    fig_dir = f'./fig_{field}/'
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir)

    ##### 3. Smilei データの抽出と一時保存
    timesteps = S.Field(0, field).getTimesteps()
    print(f"--- Step 1: {field} データの抽出と一時保存 (.npy) ---")
    
    for i in tqdm.tqdm(range(int(len(timesteps)))):
        E_raw = np.array(S.Field(0, field).getData(timestep=timesteps[i]))
        
        # 配列の次元数に合わせて安全に2次元抽出
        if E_raw.ndim == 3:
            E_matrix = E_raw[0, :, :].T
        else:
            E_matrix = E_raw.T
            
        # [V/m] から [TV/m] へ変換 (1 TV/m = 1.0e12 V/m)
        E_TVm = E_matrix * (Er * 1.0e-12)

        np.save(data_dir + f'{field}_' + str(i) + '.npy', E_TVm)

    ##### 4. 2次元カラーマップ画像の生成
    print(f"\n--- Step 2: 2次元カラーマップ画像の生成 (.png) ---")

    for i in tqdm.tqdm(range(int(len(timesteps)))):
        E_data = np.load(data_dir + f'{field}_' + str(i) + '.npy')
        
        fig = plt.figure(figsize=(15, 12))
        plt.rcParams["font.size"] = 28
        ax1 = fig.add_subplot(111)
        
        # 物理時間 [ps] の厳密な自動計算
        actual_timestep = timesteps[i]
        time_ps = (actual_timestep * dt * Tr) * 1e12
        ax1.text(0.99, 0.99, "t = {:.2f} ps".format(time_ps), va='top', ha='right', transform=ax1.transAxes, color='black')
        
        # 描画
        img1 = ax1.imshow(E_data, aspect='equal', origin='lower', cmap='bwr', vmin=vmin_val, vmax=vmax_val)
        
        # カラーバー設定
        cbar1 = fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Electric field [TV/m]")
        
        # 目盛りを綺麗な数字で5分割する (-1.0, -0.5, 0.0, 0.5, 1.0)
        cbar_ticks = np.linspace(vmin_val, vmax_val, 5)
        cbar1.set_ticks(cbar_ticks)
        cbar1.ax.set_yticklabels([f"{val:.1f}" for val in cbar_ticks])
        
        ax1.set_title(field)
        ax1.set_xlabel("x [$\mu$m]")
        ax1.set_ylabel("y [$\mu$m]")
        
        # 空間軸の目盛り指定 (nx=4096に合わせた目盛り位置)
        ax1.set_xticks([0, 1024, 2048, 3072, 4096])
        ax1.set_xticklabels(["0", "50", "100", "150", "200"])
        ax1.set_yticks([0, 1024, 2048, 3072, 4096])
        ax1.set_yticklabels(["0", "50", "100", "150", "200"])

        # 保存
        fig.savefig('{}/{}_{:03d}.png'.format(fig_dir, field, i))
        plt.close(fig)

    # 一時フォルダの削除
    shutil.rmtree(data_dir)
    print(f"\n{field} の解析が完了しました。画像は '{fig_dir}' に保存されています。")