import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 入力 dat ファイル名から、対応する出力 csv ファイル名を生成
# ------------------------------------------------------------
def make_output_name(dat_path: Path) -> Path:
    """
    入力例:
        nearshield_p100_c100_umsource_let_t_let_proton.dat

    出力例:
        spectrum_nearshield_p100_c100_umsource_let_t_let_proton.csv
    """
    # 拡張子を除いたファイル名を取得し、不要なサフィックスを削除
    stem = dat_path.stem

    # 同じディレクトリに csv を作成
    return dat_path.with_name(f"spectrum_{stem}.csv")

def main(dat_file):
    # ============================================================
    # 1. 入力ファイルの読み込み
    # ============================================================
    # datファイルを1行ずつ扱えるよう、全行をリストとして読み込む
    dat_file = Path(dat_file)
    with dat_file.open() as f:
        lines = f.readlines()

    # ============================================================
    # 2. 使用する RCF 層（iz）の定義
    # ============================================================
    # RCF2, 5層に対応する iz = 1,3
    iz_to_rcf = {
        1: "RCF2",
        3: "RCF5",
    }

    iz_list = sorted(iz_to_rcf.keys())

    # ============================================================
    # 3. 結果格納用配列の初期化
    # ============================================================
    # 各 iz（RCF層）ごとに、LET16ビン分の積算配列を用意
    spectra = {iz: np.zeros(16) for iz in iz_list}

    # LETビンの下限・上限（全セル共通なので最初の1回だけ保存）
    lwr_global = None
    upr_global = None

    # ============================================================
    # 4. datファイルを上から順に走査
    # ============================================================
    i = 0
    nlines = len(lines)

    while i < nlines:
        line = lines[i]

        # --------------------------------------------------------
        # newpage ヘッダ行の検出（ix, iy, iz を含む行）
        # --------------------------------------------------------
        if "no." in line and "ix" in line and "iy" in line and "iz" in line:
            # ix, iy, iz を正規表現で抽出
            m = re.search(r"ix\s*=\s*(\d+).*iy\s*=\s*(\d+).*iz\s*=\s*(\d+)", line)
            if not m:
                # フォーマット不一致時は次の行へ
                i += 1
                continue

            ix = int(m.group(1))
            iy = int(m.group(2))
            iz = int(m.group(3))

            # ----------------------------------------------------
            # LETテーブル（let-lwr 行）まで読み飛ばす
            # ----------------------------------------------------
            i += 1
            while i < nlines and "let-lwr" not in lines[i]:
                i += 1

            # ヘッダ行の次の行が、LETビンの数値データ
            i += 1

            # ----------------------------------------------------
            # LET 16ビン分のデータを必ず読み取る
            # ----------------------------------------------------
            lwr = np.zeros(16)   # LET下限
            upr = np.zeros(16)   # LET上限
            dose = np.zeros(16)  # 各LETビンのDose

            for k in range(16):
                parts = lines[i].split()
                lwr[k] = float(parts[0])
                upr[k] = float(parts[1])
                dose[k] = float(parts[2])
                i += 1

            # ----------------------------------------------------
            # LETビン範囲を保存（最初のセルのみ）
            # ----------------------------------------------------
            if lwr_global is None:
                lwr_global = lwr.copy()
                upr_global = upr.copy()

            # ----------------------------------------------------
            # 対象セルのみ LET スペクトルに加算
            # ----------------------------------------------------
            # ===== 元の条件（将来使う可能性があるため保持） =====
            #if iz in iz_list:
            #    if 1 <= ix <= 9:
            #        iy_min = 43 - ix
            #        iy_max = 42 + ix
            #    elif 10 <= ix <= 42:
            #        iy_min = 43 - ix
            #        iy_max = 61 - ix
            #    elif 43 <= ix <= 51:
            #        iy_min = ix - 42
            #        iy_max = 61 - ix
            #    else:
            #        continue
            #
            #    if iy_min <= iy <= iy_max:
            #        spectra[iz] += dose

            # ===== 新しい条件 =====
            # ・iz が RCF層に対応
            if iz in iz_list:
                cx, cy = 42.5, 42.5
                dx = (ix - 0.5) - cx
                dy = (iy - 0.5) - cy
                limit_w = (26 / 2) * (2 ** 0.5)
                limit_h = (56 / 2) * (2 ** 0.5)

                if abs(dx + dy) <= limit_w and abs(dx - dy) <= limit_h:
                    spectra[iz] += dose

        else:
            # newpage ヘッダ以外の行は単純にスキップ
            i += 1

    # ============================================================
    # 5. 正規化 LET スペクトル & 線量平均 LET の計算
    # ============================================================
    # 線量平均 LET 計算用の代表値（内部計算専用）
    # ※ CSV 出力には使わない
    let_rep = 0.5 * (lwr_global + upr_global)
    
    # DataFrame の列名（RCF2, RCF5）を格納するリスト
    columns = []

    # 正規化された LET スペクトルを格納するリスト
    # → 後で column_stack して DataFrame にする
    normalized_data = []

    # 各 RCF 層の線量平均 LET（LET_d）を格納するリスト
    let_d_list = []

    # ------------------------------------------------------------
    # 各 RCF 層（iz）について処理
    # ------------------------------------------------------------
    for iz in iz_list:
        columns.append(iz_to_rcf[iz])        
        
        # 三角領域内で積算された全 LET ビンの総線量
        dose_sum = spectra[iz].sum()

        if dose_sum > 0.0:
            # 各 LET ビンの線量を、全線量で割ることで正規化
            # → 「その LET が、全エネルギー損失のうち
            #     どれだけ寄与したか」を表す割合になる
            norm_spec = spectra[iz] / dose_sum

            # 線量平均 LET
            # LET_d = Σ(LET_i * Dose_i) / Σ(Dose_i)
            let_d = np.sum(let_rep * spectra[iz]) / dose_sum

        else:
            # 対象領域に陽子が全く入っていない場合の保険
            # （ゼロ割防止）
            norm_spec = np.zeros_like(spectra[iz])
            let_d = np.nan

        # 結果をリストに追加
        normalized_data.append(norm_spec)
        let_d_list.append(let_d)

    # ------------------------------------------------------------
    # DataFrame の作成
    # ------------------------------------------------------------

    # 行ラベルを「LETビン範囲」にする（棒グラフ向き）
    bin_labels = [
        f"{lwr_global[i]}-{upr_global[i]}"
        for i in range(len(lwr_global))
    ]

    data = np.column_stack(normalized_data)

    df = pd.DataFrame(data, index=bin_labels, columns=columns)
    df.index.name = "LET bin [keV/um]"

    # ------------------------------------------------------------
    # 線量平均 LET を最下行として追加
    # ------------------------------------------------------------
    df.loc["Dose-averaged LET"] = let_d_list

    # ============================================================
    # 6. CSV 出力
    # ============================================================
    out_file = make_output_name(dat_file)
    df.to_csv(out_file)

    print(f"Saved: {out_file}")

# ------------------------------------------------------------
# スクリプトとして実行された場合のエントリポイント
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_tlet.py xxx_t_let.dat")
        sys.exit(1)

    main(sys.argv[1])