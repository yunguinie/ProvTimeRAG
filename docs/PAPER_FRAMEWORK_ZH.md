# ProvTimeRAG：WSDM 2027 论文框架与跨窗口迁移说明

## 一句话论文

本文研究时间敏感 RAG 在“文本看起来相关但出处、发布者、版本或时间状态不匹配”时的可靠路由问题，并提出一个显式建模可观测发布者身份、时间/版本状态与证据不足的多任务路由器，再用保守的全局解码和固定生成契约验证其对引用质量的影响。

## 最合适的投稿定位

- 主方向：Web Mining and Content Analysis / Information Integrity。
- 辅方向：Foundation Models and Agentic Systems、Web Search。
- 不应写成：又一个通用 RAG、单纯 reranker 微调、或“所有组件都显著提升”的流水线论文。

## 三项核心贡献

1. **问题与数据契约。** 提出 publisher-visible provenance-state routing：区分访问 URL 与实际发布者身份，构造平衡 donor 的 Source-Swap，并严格控制查询、文本和 URL 泄漏。
2. **统一路由器 C2。** 在一个 cross-encoder 中联合学习 publisher、temporal/version 与 insufficiency，取得强跨域 publisher 路由，同时避免 source-only 模型在时间与拒答任务上的灾难性遗忘。
3. **从路由到可验证生成。** 冻结 C3 保守结构化解码与 C4 生成契约，证明主要收益体现在引用精确率、召回率和出处可验证性，而不是夸大答案 F1。

## 推荐章节结构（WSDM 主会 9 页正文）

1. **Introduction（1.1–1.3 页）**：真实 Web 证据的来源漂移；传统相关性 reranking 的失败；三项贡献。
2. **Related Work（0.8–1.0 页）**：RAG/出处、时间知识、事实核验、结构化证据选择。
3. **Problem Formulation（0.6 页）**：原子 claim、候选证据、publisher/document/time state、abstention、bundle。
4. **Method（2.0–2.3 页）**：元数据契约；C2 多任务目标；C3 保守解码；C4 接口。
5. **Experimental Setup（1.1–1.3 页）**：数据、泄漏控制、外部集、基线、指标、统计检验。
6. **Results（1.5–1.8 页）**：主表、外部泛化、消融、C3/C4。
7. **Analysis and Limitations（0.7 页）**：seen/unseen、URL 消融、C3 不显著、C1 上限、API failure policy。
8. **Conclusion（0.25 页）**。
9. **Ethical Considerations**：数据许可、隐私、API 与人工标注、失败风险。

## 写作红线

- 不把 C3 的 +0.35 pp 写成显著提升。
- 不说多任务显著提升 publisher；应说“保留接近 source-only 的 publisher 性能，同时统一解决另外两项任务”。
- 不把 C1 的 0.48 row exact 描述成成熟 SOTA。
- 不混用 legacy v1、gold-first 队列或 donor-collapse 外部集。
- 不用“0.9 才能发”做跨任务比较；bundle exact、group Top-1、citation metrics 的难度和定义不同。

## 当前是否可以开始写作

可以立即开始正文。数据匹配的 source-only 对照如果尚未同步，只影响一行消融表和一句结论，不阻塞引言、相关工作、方法、数据契约和绝大多数实验章节。最终英文版必须使用 ACM `sigconf,anonymous,review` 模板并压到官方页数。
