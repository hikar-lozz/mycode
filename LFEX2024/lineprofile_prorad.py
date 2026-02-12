import numpy as np
import pandas as pd
import sys

# -----------------------------
# コマンドライン引数
# -----------------------------
if len(sys.argv) < 2:
    print("使い方: python script.py particles.csv")
    sys.exit(1)

csv_file = sys.argv[1]

# -----------------------------
# CSV読み込み（粒子座標）
# -----------------------------
data = np.loadtxt(csv_file, delimiter=",")
x = data[:, 0]
y = data[:, 1]

# -----------------------------
# メッシュ設定
# -----------------------------
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
nx, ny = 600, 600

# -----------------------------
# 範囲内粒子抽出
# -----------------------------
mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
x = x[mask]
y = y[mask]

# -----------------------------
# 粒子数メッシュ（= 1層RCF）
# -----------------------------
H, _, _ = np.histogram2d(
    y_max - y, x,
    bins=[ny, nx],
    range=[[0, y_max - y_min], [x_min, x_max]]
)
deposit_array = H.astype(float)   # 600×600

# -----------------------------
# near shield 条件
# -----------------------------
length_take_average = 72
length_line = 358

# -----------------------------
# ラインプロファイル計算
# -----------------------------
profile = []

for j in range(-length_line // 2, length_line // 2 + 1):
    xj = 300 + j
    yj = 300 + j

    values = []
    for k in range(-length_take_average // 2, length_take_average // 2 + 1):
        xk = xj + k
        yk = yj - k

        # 配列範囲チェック（安全策）
        if 0 <= xk < 600 and 0 <= yk < 600:
            values.append(deposit_array[int(xk), int(yk)])

    profile.append(np.nanmean(values))

profile = np.array(profile)

# -----------------------------
# 保存
# -----------------------------
df = pd.DataFrame({"LineProfile": profile})
output_filename = 'lineprofile_' + csv_file.replace('.csv', '.xlsx')
df.to_excel(output_filename, index=False)
print(f"near shield のラインプロファイルを {output_filename} に保存しました")