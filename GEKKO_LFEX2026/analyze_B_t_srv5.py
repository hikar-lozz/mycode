# hil5利用時はimport happiをコメントアウトして以下を実行
diagnostics_path = "/media/ymraid5-5/Kensei/Smilei/scripts/Diagnostics.py"
exec(compile(open(diagnostics_path).read(), diagnostics_path, 'exec'))

import math
# import happi
import numpy as np
import os
import tqdm
import scipy.constants
from matplotlib import pyplot as plt

##### 1. 物理定数とインプットファイルからのパラメータ設定
e   = scipy.constants.e               # 電荷 [C]
me  = scipy.constants.m_e             # 電子質量 [kg]
c   = scipy.constants.c               # 光速 [m/s]

# レーザーと基準値の設定
lr  =  1.053e-6                       # レーザー波長 [m]
wr  =  2. * math.pi * c / lr          # 角周波数 [rad/s]
Tr  =  1. / wr                        # 基準時間 [s]
Lr  =  c / wr                         # 基準長さ [m]
Br  =  me * wr / e                    # 基準磁場 [T]

# インプットファイルから抽出した空間・時間パラメータ
um = 1.e-6 / Lr                       # 1 um の正規化単位値
dx_um = 0.048828125                   # 実際の dx (um単位)
dx = dx_um * um                       # セルサイズ x
dy = dx_um * um                       # セルサイズ y

# 2D Cartesian CFL条件から正規化時間刻み dt を計算
dt_cfl = 1.0 / math.sqrt(1.0 / (dx**2) + 1.0 / (dy**2))
dt = 0.99 * dt_cfl                    # timestep_over_CFL = 0.99

##### 2. フォルダの準備
fig_dir = './fig_Bz_time_evolution/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

##### 3. Smilei データの読み込みと解析領域（マスク）の自動生成
S = happi.Open('./')
timesteps = S.Field(0, 'Bz').getTimesteps()

# 最初のステップ(t=0)のデータを読み込み、実際の配列サイズ（4097x4097など）を取得する
sample_Bz_raw = np.array(S.Field(0, 'Bz').getData(timestep=timesteps[0]))
if sample_Bz_raw.ndim == 3:
    sample_Bz_matrix = sample_Bz_raw[0, :, :].T
else:
    sample_Bz_matrix = sample_Bz_raw.T

ny_actual, nx_actual = sample_Bz_matrix.shape

# シミュレーションボックスの厳密な中心インデックス
cx = nx_actual / 2.0
cy = ny_actual / 2.0

# 半径 50.0 um に相当するセル数を計算
r_um = 50.0
r_cell = r_um / dx_um

# 実際の配列サイズに基づく2次元グリッドマスクを作成
x_indices = np.arange(nx_actual)
y_indices = np.arange(ny_actual)
X, Y = np.meshgrid(x_indices, y_indices)
mask = (X - cx)**2 + (Y - cy)**2 < r_cell**2

##### 4. データ抽出
# プロット用データ格納リスト
time_ps_list = []
bz_max_list = []
bz_mean_list = []

print("--- Step 1: Bz データの抽出と解析 ---")
for i in tqdm.tqdm(range(int(len(timesteps)))):
    Bz_raw = np.array(S.Field(0, 'Bz').getData(timestep=timesteps[i]))
    
    # 配列の次元数に合わせて安全に2次元抽出
    if Bz_raw.ndim == 3:
        Bz_matrix = Bz_raw[0, :, :].T
    else:
        Bz_matrix = Bz_raw.T
        
    Bz_kT = Bz_matrix * (Br * 0.001)  # [T] から [kT] へ変換
    Bz_abs = np.abs(Bz_kT)            # 磁場強度の絶対値を取得
    
    # 動的に生成したマスクを用いてデータを抽出
    Bz_target_region = Bz_abs[mask]
    
    # 最大値と平均値を計算してリストに追加
    bz_max_list.append(np.max(Bz_target_region))
    bz_mean_list.append(np.mean(Bz_target_region))
    
    # 物理時間 [ps] の計算
    actual_timestep = timesteps[i]
    time_ps = (actual_timestep * dt * Tr) * 1e12
    time_ps_list.append(time_ps)

##### 5. グラフの作成と保存
print("\n--- Step 2: グラフの生成 (.png) ---")

fig = plt.figure(figsize=(12, 8))
plt.rcParams["font.size"] = 22
ax = fig.add_subplot(111)

# 最大値と平均値をプロット
ax.plot(time_ps_list, bz_max_list, label='Maximum $|B_z|$', color='red', linewidth=2.5)
ax.plot(time_ps_list, bz_mean_list, label='Mean $|B_z|$', color='blue', linewidth=2.5, linestyle='--')

# 軸とタイトルの設定
ax.set_title(r'Time Evolution of $|B_z|$ ($r < 50\,\mu\mathrm{m}$)')
ax.set_xlabel('Time [ps]')
ax.set_ylabel('Magnetic field strength [kT]')

# ★変更点: 横軸の余白をなくす ＆ 7の目盛りを確実に表示させるための処理
t_min = time_ps_list[0]
t_max = time_ps_list[-1]

# 描画範囲を最初から最後までぴったりに設定（丸め誤差で端の目盛りが消えないよう小数第3位で丸める）
ax.set_xlim(t_min, round(t_max, 3))

# 0から最後の時間まで、1 ps刻みで目盛りを明示的に設定
max_tick = math.ceil(t_max)
ax.set_xticks(np.arange(0, max_tick + 1, 1))

ax.legend(loc='upper right')

# 保存
save_path = os.path.join(fig_dir, 'Bz_time_evolution.png')
fig.savefig(save_path, bbox_inches='tight')
plt.close(fig)

print("\n解析が完了しました。画像は '{}' に保存されています。".format(save_path))