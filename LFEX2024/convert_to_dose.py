import sys
import cv2
import numpy as np

# TIFF画像を読み込む
filename = sys.argv[1]
image_array = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

# 画像の各ピクセルの青成分だけを取り出す
blue_array = image_array[:, :, 0]

# 配列の形状を確認
print("Blue array shape:", blue_array.shape)
print("Blue array:", blue_array[:5, :5])

# ODを計算する関数を定義
def calculate_od(I_film):
    gamma = 0.5     # Blue channel
    I_min = 1600
    I_max = 65500
    numerator_array = (I_film**(1/gamma)) - (I_min**(1/gamma))
    denominator_array = (I_max**(1/gamma)) - (I_min**(1/gamma))
    ratio_array = numerator_array / denominator_array
    ratio_array = np.where(ratio_array > 0, ratio_array, np.nan)      # 比率が0または負の値の場合、NaNを設定
    od_array = -np.log10(ratio_array)
    return od_array

# 全てのピクセル値に対してODを計算し、新しい配列を作成
od_array = calculate_od(blue_array)

# netODの定義
od_unexp = 0.20     # RCFの端っこにおけるODを代入
netod_array = od_array - od_unexp

# Dを計算する関数を定義
# (netOD*c5 - c2)D**2 + (netOD*c4 - c1)D + netOD * c3 = 0 を解く
def calculate_d(netOD):
    c1 = 1.937e6
    c2 = 0.379e6
    c3 = 4.279e6
    c4 = 4.491e6
    c5 = 0.204e6

    a_array = netOD * c5 - c2
    b_array = netOD * c4 - c1
    c_array = netOD * c3
    discriminant_array = b_array**2 - 4*a_array*c_array

    # 初期値をnanにしておく
    d_array = np.full_like(netOD, np.nan, dtype=np.float64)

    # a_array == 0 かつ　b != 0のケース（一次方程式）
    dimention_1 = (a_array == 0) & (b_array != 0)
    D_1 = -c_array[dimention_1] / b_array[dimention_1]
    d_array[dimention_1] = np.where(D_1 >= 0, D_1, np.nan)

    # a_array != 0 かつ discriminant_array >= 0 のケース（二次方程式）
    dimention_2 = (a_array != 0) & (discriminant_array >= 0)
    D1 = (-b_array[dimention_2] + np.sqrt(discriminant_array[dimention_2])) / (2 * a_array[dimention_2])
    D2 = (-b_array[dimention_2] - np.sqrt(discriminant_array[dimention_2])) / (2 * a_array[dimention_2])
    D1 = np.where(D1 >= 0, D1, np.nan)
    D2 = np.where(D2 >= 0, D2, np.nan)
    d_array[dimention_2] = np.fmax(D1, D2)      # 2つの解の内大きい方を採用する

    return d_array

# 全てのピクセル値に対してDを計算し、新しい配列を作成
d_array = calculate_d(netod_array)

# Dの配列を表示
print("D array shape:", d_array.shape)
print("D array:", d_array[:5, :5])

# d_arrayを保存
output_filename = 'dose_' + filename.replace('.tif', '.npz')
np.savez(output_filename, d_array = d_array)
print(f"d_arrayを{output_filename}として保存しました")