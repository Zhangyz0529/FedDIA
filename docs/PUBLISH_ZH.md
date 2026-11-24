# 如何公开本精简版

## 已保留

核心模型与训练/缓存/外部方法适配实现；正式逐seed结果和必要汇总；验证候选和选择；核心通信账本；Protocol B/C与桥接结果；适配来源与关键修复说明；五图与绘图数据。逐文件见PUBLIC_FILES.csv。

## 已移除

所有重复code快照、交付报告、WORK_LOG/CURRENT_PROGRESS等过程状态、旧版与理论复用预算图、重复验收PASS文件、旧版论文增补片段和与本轮无关的审查脚本。旧固定日程图也未保留，但结果未丢弃。原完整公开候选包仍可供作者追溯。

## 发布前补齐

1. 作者、正式论文标题/引用与CITATION.cff。
2. 作者批准的LICENSE，以及适配衍生代码涉及的第三方通知；未自行假定MIT。
3. 处理后协议数据、固定ID、预处理参数、权重/预测等独立资产的真实版本、链接与哈希；参考DATA_AVAILABILITY.md。
4. 解决历史数据路径和缺失依赖，在新环境确认训练入口；当前只验证了预算图重建，不能把它等同于完整训练复现。
5. 将论文每表每图对应到正式结果；Re-Fed新版10/6/6与旧结果须明确区分，不把旧external表当作最新执行复用表。

建议先在GitHub创建空仓库FedDIA。将本文件夹内容作为仓库根目录，完成上述信息后执行（OWNER为你的账号）：

```sh
git init -b main
git add .
git status --short
git commit -m "Release curated FedDIA artifacts"
git remote add origin https://github.com/OWNER/FedDIA.git
git push -u origin main
```

作者自行确认公开可见性并创建版本Release。本次仅整理本地包，没有创建仓库或推送。主仓库提交解压后的文件，ZIP可作为发布下载附件。大型数据和权重放独立存档，并给出永久链接。无需为公开新增实验。

GitHub官方步骤：https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=linux
