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
e  = scipy.constants.e               # 電荷 [C]
me = scipy.constants.m_e             # 電子質量 [kg]
c  = scipy.constants.c               # 光速 [m/s]

# レーザーと基準値の設定
lr  =  1.053e-6                      # レーザー波長 [m]
wr  =  2. * math.pi * c / lr         # 角周波数 [rad/s]
Tr  =  1. / wr                       # 基準時間 [s]
Lr  =  c / wr                        # 基準長さ [m]
Br  =  me * wr / e                   # 基準磁場 [T]

# インプットファイルから抽出した空間・時間パラメータ
um = 1.e-6 / Lr                      # 1 um の正規化単位値
dx = 0.048828125 * um                # セルサイズ x
dy = 0.048828125 * um                # セルサイズ y
nx = 8192                            # x方向セル数
ny = 8192                            # y方向セル数

# 2D Cartesian CFL条件から正規化時間刻み dt を計算
dt_cfl = 1.0 / math.sqrt(1.0 / (dx**2) + 1.0 / (dy**2))
dt = 0.99 * dt_cfl                   # timestep_over_CFL = 0.99
diag_every = 200                     # DiagFields(every = 200)

##### 2. フォルダの準備
data_dir = './Bz_data/'
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

fig_dir = './fig_Bz/'
if os.path.exists(fig_dir):
    shutil.rmtree(fig_dir)
os.makedirs(fig_dir)

##### 3. Smilei データの読み込みと一時保存
S = happi.Open('./')
timesteps = S.Field(0, 'Bz').getTimesteps()

print("--- Step 1: Bz データの抽出と一時保存 (.npy) ---")
for i in tqdm.tqdm(range(int(len(timesteps)))):
    Bz_raw = np.array(S.Field(0, 'Bz').getData(timestep=timesteps[i]))
    Bz_matrix = Bz_raw[0, :, :].T
    Bz_kT = Bz_matrix * (Br * 0.001)  # [T] から [kT] へ変換
    np.save(data_dir + 'Bz_' + str(i) + '.npy', Bz_kT)

##### 4. 2次元カラーマップ画像の生成
print("\n--- Step 2: 2次元カラーマップ画像の生成 (.png) ---")
for i in tqdm.tqdm(range(int(len(timesteps)))):
    Bz_data = np.load(data_dir + 'Bz_' + str(i) + '.npy')
    
    fig = plt.figure(figsize=(15, 12))
    plt.rcParams["font.size"] = 28
    ax1 = fig.add_subplot(111)
    
    # 物理時間 [ps] の厳密な自動計算
    actual_timestep = timesteps[i]
    time_ps = (actual_timestep * dt * Tr) * 1e12
    ax1.text(0.99, 0.99, "time = {:.2f} [ps]".format(time_ps), va='top', ha='right', transform=ax1.transAxes, color='black')
    
    # 描画
    img1 = ax1.imshow(Bz_data, aspect='equal', origin='lower', cmap='bwr', vmin=-30, vmax=30)
    
    #【ここを修正しました】make_axes_locatable を使わずにカラーバーを設定
    # fraction=0.046 でメインのグラフと縦幅を揃え、pad=0.04 で適切な隙間を空けています
    cbar1 = fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Magnetic field [kT]")
    
    ax1.set_title('Bz')
    ax1.set_xlabel("x [$\mu$m]")
    ax1.set_ylabel("y [$\mu$m]")
    
    # 空間軸の目盛りを自動マッピング
    ax1.set_xticks([0, 2048, 4096, 6144, 8192])
    ax1.set_xticklabels(["0", "100", "200", "300", "400"])
    ax1.set_yticks([0, 2048, 4096, 6144, 8192])
    ax1.set_yticklabels(["0", "100", "200", "300", "400"])

    # 保存
    fig.savefig('{}/Bz_{:03d}.png'.format(fig_dir, i))
    plt.close(fig)  # メモリリーク対策のため fig を明示的に閉じるように変更

# 一時フォルダの削除
shutil.rmtree(data_dir)
print("\n解析が完了しました。画像は '{}' に保存されています。".format(fig_dir))