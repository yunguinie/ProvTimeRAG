# ProvTimeRAG 当前交接（2026-08-10）

## 论文主线

ProvTimeRAG 研究时间敏感 Web RAG 的出处状态路由：候选文本即使语义相关，也可能来自错误发布者、错误文档版本或错误时间状态。主方法 C2 显式读取 publisher-visible URL/domain，并联合学习 publisher、temporal/version 与 insufficiency。

## 已冻结、可写入论文的结果

- C2 clean blind：0.8881 ± 0.0060 Top-1（828 groups）。
- C2 FinFact：0.9383 ± 0.0059 Top-1（11,164 groups）。
- 最强冻结 OTS 基线 Qwen3-Reranker-0.6B：clean blind 0.7343，FinFact 0.8793。
- no-URL 消融在 dev / clean blind / FinFact 分别下降 17.75 / 20.17 / 15.16 pp。
- C3 FinFact bundle exact：0.7686 → 0.7721，平均 +0.35 pp，但不显著，不能写成主要胜利。
- C4 fair-order 678：citation P 0.4998 → 0.5470，citation R 0.8916 → 0.9400；answer F1 差异不显著。
- C1 独立人工金标最佳 row exact 0.48，仅作为元数据恢复可行性与数据治理证据。

## 仍待同步

服务器上的 data-matched source-only 对照尚未同步到本地。它只用于补充区分“多任务效应”和“数据量效应”，不阻塞论文主体写作。不要猜测或手填其数值。

## 禁止使用的旧结果

- legacy Source-Swap v1 URL contract；
- AVerImaTeC donor-collapse v1；
- gold-first/non-canonical C4 队列；
- metricfix 之前约 0.36–0.41 的早期 C3 主结果；
- 任何测试后重选阈值或重校准的外部结果。

## 写作入口

- 中文完整初稿：`paper/ProvTimeRAG_WSDM2027_中文论文初稿_v1.docx`
- 论文框架：`docs/PAPER_FRAMEWORK_ZH.md`
- 冻结结果卡：`docs/RESULTS_CARD.md`
- 复现说明：`docs/REPRODUCIBILITY.md`
- 机器可读结果：`results/release/main_results.json`

## 下一步顺序

1. 同步 data-matched source-only 结果并更新一行消融结论。
2. 把中文初稿翻译/压缩到 ACM `sigconf,anonymous,review` 英文 9 页正文。
3. 生成主图、主表和 appendix；核对所有数字与 SHA-256。
4. 配置 GitHub 远程仓库并推送当前安全提交。
5. 在提交前完成匿名化、伦理声明、引用与 artifact checklist。
