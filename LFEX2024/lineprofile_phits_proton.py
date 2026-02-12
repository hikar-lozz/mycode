import sys
import numpy as np
import pandas as pd

# t_deposit_radiographyを読み込む
filename = sys.argv[1]

# ラインプロファイルを取るためにシールドの位置を確認
print("操作： 実験構成におけるシールドの配置を教えてください.")
user_input = input("position of shield 'far' or 'near': ")
# 実験のRCF(35×35mm,600dots per inch)とシミュレーション(3×3cm,600×600cell=20cells per mm)
# 上記を考慮してプロファイルを取る長さを決定
if user_input.lower() == "far":
    length_take_average = 16
    length_line = 84
elif user_input.lower() == "near":
    length_take_average = 50
    length_line = 254

# RCF10層分のラインプロファイルを格納する配列を定義
lineprofile_array = np.full((length_line + 1, 10), np.nan)

# 以下の操作をRCF10層分だけ繰り返す
for i in range(1,11):
    start_line = 303 + 72058 * (i - 1)
    end_line = 36302 + 72058 * (i - 1)
    with open(filename, 'r') as file:
        lines = file.readlines()[start_line-1:end_line]
    deposit_array = np.array([list(map(float, line.split())) for line in lines])
    deposit_array = deposit_array.reshape((600, 600))       # output sizeの600×600に合わせる
    profile = []        # 繰り返し使えるようにここで初期化
    # ある層のRCFにおいて, プロファイルをとる方向のラインを辿るように繰り返す
    for j in range(-length_line // 2, length_line // 2 + 1):
        xj = 300 + j
        yj = 300 + j
        values = []     # 繰り返し使えるようにここで初期化
        # プロファイルをとる方向のライン上のある点を中点とする, 平均をとる方向のラインを辿るように繰り返す
        for k in range(-length_take_average // 2, length_take_average // 2 + 1):
            xk = xj + k
            yk = yj - k
            # 平均をとる方向のラインにおけるdepositを取得
            values.append(deposit_array[int(xk), int(yk)])
        # プロファイルをとる方向のラインにおけるdepositの平均を取得
        profile.append(np.nanmean(values))
    # RCF10層分のラインプロファイルを取得
    lineprofile_array[:, i-1] = profile

# lineprofile_arrayをわかりやすい形で保存するためにカラム名を付けて保存
column_names = [f"Layer {i}" for i in range(1, 11)]
df_lineprofile = pd.DataFrame(lineprofile_array, columns=column_names)
output_filename = 'lineprofile_' + filename.replace('.dat', '.xlsx')
df_lineprofile.to_excel(output_filename, index=False)
print(f"lineprofile_arrayを{output_filename}として保存しました")