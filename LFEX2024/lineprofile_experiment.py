import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# d_arrayを読み込む
filename = sys.argv[1]
loaded_data = np.load(filename)
d_array = loaded_data['d_array']

# 今後の操作を示す
print("操作： 今から表示するマップにおいて, ピンホールとナイフエッジの交点を結ぶ線分の中点の座標(計10点)をメモしてください.")
print("注意： ただし画像の横がx軸, 縦がy軸, 原点は左上であることに気を付けてください.")
assert input("内容を確認したらokと入力してください：").lower() == 'ok', "確認できませんでしたので処理を中断します."

# d_arrayをカラーマップに表示（発表用の図）
min_value = np.nanmin(d_array)
max_value = 10      # [kGy]
plt.imshow(d_array, cmap='plasma', vmin=min_value, vmax=max_value)
plt.colorbar()
plt.title('Dose [kGy]')
plt.xticks([])
plt.yticks([])
plt.show()

# 取得した座標を入力させる
point_array = np.zeros((10, 2), dtype=float)
for i in range(10):
    user_input = input(f"{i+1}層目RCFの座標をx,y(スペースなし)の形式で入力してください（例: 25.0,40.2）：")
    y, x = map(float, user_input.split(","))
    point_array[i] = [x, y]

# ラインプロファイルを取るためにシールドの位置を確認
print("操作： 実験構成におけるシールドの配置を教えてください. shot5733-5746はfar, shot5750-5761はnearです.")
user_input = input("position of shield 'far' or 'near': ")
if user_input.lower() == "far":
    length_take_average = 20
    length_line = 100
elif user_input.lower() == "near":
    length_take_average = 60
    length_line = 300

# RCF10層分のラインプロファイルを格納する配列を定義
lineprofile_array = np.full((length_line + 1, 10), np.nan)

# 以下の操作をRCF10層分だけ繰り返す
for idx, (x_center, y_center) in enumerate(point_array):
    profile = []        # 繰り返し使えるようにここで初期化
    # ある層のRCFにおいて, プロファイルをとる方向のラインを辿るように繰り返す
    for i in range(-length_line // 2, length_line // 2 + 1):
        xi = x_center + i
        yi = y_center + i
        values = []     # 繰り返し使えるようにここで初期化
        # プロファイルをとる方向のライン上のある点を中点とする, 平均をとる方向のラインを辿るように繰り返す
        for j in range(-length_take_average // 2, length_take_average // 2 + 1):
            xj = xi + j
            yj = yi - j
            # 平均をとる方向のラインにおけるDoseを取得
            values.append(d_array[int(xj), int(yj)])
        # プロファイルをとる方向のラインにおけるDoseの平均を取得
        profile.append(np.nanmean(values))
    # RCF10層分のラインプロファイルを取得
    lineprofile_array[:, idx] = profile

# lineprofile_arrayをわかりやすい形で保存するためにインデックス名とカラム名を付けて保存
column_names = [f"Layer {i}" for i in range(1, 11)]
df_lineprofile = pd.DataFrame(lineprofile_array, columns=column_names)
i = np.arange(length_line + 1)
index_values = (i - (length_line / 2)) * ((2.54 / 600) * 10 * np.sqrt(2))
df_lineprofile.index = index_values
df_lineprofile.index.name = 'Length [mm]'
output_filename = 'lineprofile_' + filename.replace('.npz', '.xlsx')
df_lineprofile.to_excel(output_filename, index=True)
print(f"lineprofile_arrayを{output_filename}として保存しました")