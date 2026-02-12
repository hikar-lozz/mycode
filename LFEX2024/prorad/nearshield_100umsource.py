# -*- coding: utf-8 -*-
# layer1-5=[(3.0050, 2.5), (3.0159, 3.9), (3.0268, 5.1), (3.0377, 6.1), (3.0486, 6.9)]
# layer6-10=[(3.0595, 7.8), (3.0704, 8.5), (3.0813, 9.2), (3.0922, 9.7), (3.1031, 10.3)]

fformat = 'analytic_grid'           # 使用するグリットの種類
fname = ''                          # データファイルを必要とするグリットの場合、そのファイルのパス
r_source = 0.0050                   # プロトン源の有効半径（cm）
source_loc = [0.0, 0.0, 0.0]        # プロトン源の位置（cm）
prop_dir = [0.0, 0.0, 1.0]          # プロトンが発射される方向。ベクトルは自動で正規化される。
film_loc = [0.0, 0.0, 3.0486]       # フィルムの中心位置（cm）
film_axis1 = [-1.75, 0.0, 0.0]      # フィルムの第1軸（cm）
film_axis2 = [0.0, 1.75, 0.0]       # フィルムの第2軸（cm）
NP = 50000000                       # 発射するプロトンの数
ntraces = 0                         # 軌跡を記録するプロトンの数
E0 = 6.9                            # 初期プロトンエネルギー（MeV）
l_s2start = 0.02                    # プロトンはprop_dir方向にl_s2start（cm）進んだ位置にあるディスク上から発射される
spread_angle = 45.0                 # プロトンの発射半角（°）
l_prop = 0.6                        # ディスク面からprop_dir方向にl_prop（cm）進むまでシミュレーションを行う
nsteps = 500                        # 1粒子あたりのステップ数

hist_maxfluence = 800               # 飛跡カラーマップの最大値
plot_fluence = True                 # 飛跡カラーマップ
plot_traces = False                 # 軌跡マップ
plot_quiver = False                 # 電磁場マップ
save_images = True                  # マップの自動保存
fileout = 'raw_xy.csv'              # 生データの出力

ngridx = 600                        # x, y, z方向のセル数
ngridy = 600
ngridz = 100
lx = 0.4                            # x, y, z方向の空間サイズ（cm）
ly = 0.4
lz = l_prop
gridcorner = [-lx/2.0, -ly/2.0, 0]  # グリッドの左下奥の座標
from E_fields import fields         # 引数coord = (x, y, z)に対して9要素の場の値を返す関数
grid_nthreads = 4                   # （任意）Pythonの並列化を利用してグリッド初期化を高速化する