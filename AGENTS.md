# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概觀

Zero-shot 文字分類（情緒分析）RESTful 伺服器：FastAPI + HuggingFace `zero-shot-classification` pipeline（NLI 模型，預設 `asadfgglie/mDeBERTa-v3-base-xnli-multilingual-zeroshot-v1.1`）。沒有測試與 lint 設定。

## 指令（Windows）

- 首次安裝：`setup.bat`（建立 `venv` 並安裝 `requirements.txt`，torch 為 CUDA 12.1 版）
- 啟動伺服器：`start.bat`（等同 `venv\Scripts\python.exe api.py`），監聽 `0.0.0.0:20823`，API 文件在 `http://localhost:20823/docs`
- 速度測試：先啟動伺服器，再執行 `venv\Scripts\python.exe speed_benchmark.py`（對 `localhost:20823/analyze` 發請求，使用 HF dataset `asadfgglie/BanBan_2024-10-17-facial_expressions`，需要 `return_testing_data` 回傳的計時欄位）

## 架構

- `api.py`：兩個 endpoint：`POST /analyze`（原始、向後相容）與 `POST /v1/systemone`（見下）。**模型只在 `if __name__ == '__main__'` 內載入**並賦值給模組層級的 `pipe`，因此必須以 `python api.py` 啟動；直接用 `uvicorn api:app` 會因 `pipe` 為 `None` 而失敗。載入流程包含 fp16、CUDA、可選 `torch.compile(backend="cudagraphs")` 與一次 warm-up。
- `schema.py`：pydantic 請求/回應模型。`AnalyzeRequest` 以 `request.model_dump()` 直接展開成 `pipe(**...)` 的參數，所以欄位名稱必須與 pipeline 參數一致；`weights` 設為 `Field(exclude=True)`，就是為了不被傳進 pipeline。
- `config.py`：全域開關（`USE_TRANSLATOR`、`USE_TORCH_COMPILE`、`MODEL_NAME`、`MODEL_DTYPE`），在 import 時讀取。
- `weights` 後處理（`rerank_by_weight`，在 `api.py` 內）：pipeline 輸出後，依 label 對應的權重縮放分數；`multi_label=False` 時重新正規化，`True` 時乘上權重總和，最後重新排序 labels/scores。
- 翻譯：`USE_TRANSLATOR=True` 時 `api.py` 使用 `googletrans` 先翻譯輸入（`googletrans` 不在 `requirements.txt`）；`tool.py` 的 `FasterMyMemoryProvider`（`translate` 套件、session 重用）目前未被 `api.py` 引用。
- 回應中的 `sequence` 會被改回翻譯前的原文；`return_testing_data=True` 時回傳 `AnalyzeTestResponse`（含推論/翻譯/總時間與模型設定），供 `speed_benchmark.py` 使用。
- `POST /v1/systemone`（`system_one`）：API 設計參考 TypeSafe 的 `systemone`，但**不會呼叫 TypeSafe**，每個 question 都會被轉成一次 `AnalyzeRequest` 並直接 `await analyze(...)`，所以翻譯與 `weights` 邏輯都共用 `/analyze`。對應關係：`choice` → `multi_label=False`；`score` → `multi_label=True`（依機率排序）；`noul` → 兩個標籤（`instructions.format(criteria['true'/'false'])`）的 `multi_label=True`，`noul` 欄位取 true 標籤的分數，`confidence` 是兩個標籤中較大分數除以兩者總和。`choice` 的標籤若有描述，會以 `label:description` 送進模型，並以該字串回傳。`model` 欄位只為相容，實際一律用 `config.MODEL_NAME`。請求/回應模型在 `schema.py` 的 `SystemOne*` / `*Question`。
