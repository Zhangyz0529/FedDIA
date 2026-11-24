# FedDIA — 精简公开候选版

这是2026-11-24整理的研究公开候选包。**当前仓库保留了核心实现、数据管线代码及相关文档。**

## ⚠️ 当前版本状态说明
因 GitHub 网页端单次上传文件数量限制，**本仓库暂未包含 `supplementary_data/` 目录**（该目录包含原实验批次证据路径、逐 seed 结果、图表输入数据等）。因此，本仓库目前仅作为**核心代码与文档的公开预览版**。完整的实验数据与图表重现环境暂需依赖本地完整包。

## 目录说明

- **根目录 Python 文件与 `data_pipeline/`**：核心算法实现及数据处理代码。
- **`docs/`**：公开步骤、数据获取说明与授权状态。
- **`tools/verify_release.py`**：完整性校验脚本（*注：运行需依赖缺失的 supplementary_data 目录*）。
- **`supplementary_data/`**：*(当前未上传，计划后续补充)* 按原实验批次保留证据路径，避免修改源码和证据绑定。

## 关于“可直接验收”
原完整包提供以下验收命令，但**在当前精简版仓库中，因缺失 `supplementary_data` 目录下的输入数据，运行校验和绘图脚本会因找不到输入文件而报错**。

```sh
# 以下命令需在拥有完整 supplementary_data 的本地环境中运行
pip install -r requirements-figures.txt
python tools/verify_release.py
python plot_selected_policy_20261121.py
