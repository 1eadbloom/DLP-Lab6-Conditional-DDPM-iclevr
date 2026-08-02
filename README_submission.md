# Lab6 繳交說明（109205057 徐祥智）

## 什麼是 iclevr 資料？

**i-CLEVR** 是本實驗專用的合成影像資料集，需從課程 **e3 / NTU COOL** 或助教提供的 **Google Drive** 下載 **`iclevr.zip`**（與 `file.zip` 分開）。

解壓後目錄結構應為：

```
Lab6/
  iclevr/
    CLEVR_train_000051_0.png
    CLEVR_train_000051_1.png
    ...
```

- **影像內容**：類似 CLEVR 的 3D 合成場景（球、立方體、圓柱 × 8 種顏色），每張 1～3 個物體。
- **與 train.json 的關係**：`train.json` 的 key 是檔名、value 是物體標籤列表，共 **18009** 張訓練圖；檔名必須與 `iclevr/` 內 PNG 一致。
- **解析度**：原始約 32×32，評估器使用 **64×64**（與生成目標一致）。
- **用途**：訓練你的 conditional DDPM；沒有這包圖，只能用程式化備援，分類準確率通常很低。

`file.zip` 內另有：`objects.json`、`train.json`、`test.json`、`new_test.json`、`evaluator.py`、`checkpoint.pth`（**不含**訓練影像）。

## 已產生檔案

| 檔案 | 說明 |
|------|------|
| `lab6_conditional_ddpm.py` | 原始碼 |
| `DL_LAB6_report.pdf` | 繁體中文實驗報告 |
| `images/test/`、`images/new_test/` | 各 32 張 PNG |
| `DL_LAB6_109205057_徐祥智.zip` | 繳交壓縮檔 |

## 使用官方 iclevr 重新訓練（建議）

```powershell
cd c:\Users\user\Downloads\Lab6
# 將 iclevr.zip 解壓到 .\iclevr\
python lab6_conditional_ddpm.py --mode train --epochs 50 --max-train-samples 18009
python lab6_conditional_ddpm.py --mode all --student-id 109205057 --student-name 徐祥智
```
