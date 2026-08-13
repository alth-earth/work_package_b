# 工作包 B 的 AI 开发约束

本文件由 `../work_package_b_handoff/AGENTS.md` 复制并针对正式 B 工程落地。若本文件与
当前共享/A/C 公共合同冲突，以当前代码、Schema 和生产者—消费者测试的严格交集为准。

## 边界

- B 只消费 A 公共 `PreparedWindow`/`DatasetBundle`/`StandardDataFrame` 与共享
  `RunContext`，发布 C 公共 `bc.risk-frame.v2` 和 committed window。
- 禁止下载原始资料、读取 A SQLite/私有目录，禁止生成路线、最终航速、ETA 或 CD 产物。
- 只导入 A、共享和 C 的公共合同，不导入 C 的 planner/risk/grid/cost/replanning 内部实现。
- 当前规则必须命名 `demo_unvalidated`，不得声称科学校准或可用于真实导航。

## 不变量

- 所有时间必须为 UTC；每个来源 `issue_time <= as_of_time`。
- `RunContext`、经共享包复核的 `a.dataset-bundle.v2`、实际 A frames 必须逐项同一身份。
- 正式完整窗严格为 60 分钟闭区间；未知风险不得填 0。
- `model_config_digest` 绑定 B 网格策略、时间策略和规则，不绑定某条走廊的 bbox/坐标。
- `source_valid_mask` 不是 `land_sea_mask`；首版 hard mask 只由正式 land/sea 分类产生。
- `risk_level = min(5, floor(risk_score * 5) + 1)`；未知点保守为 5。
- 发布对象内容寻址、不可变；旧 generation 晚到不得发布；窗口 manifest 原子提交。

## 工程与验收

- Python 3.13；Mamba 管理 Python/原生库，uv 管理 Python 依赖和 `uv.lock`。
- 编辑只限本工程；跨包变化由各包独立实现并以公共接口集成。
- 每次至少执行：`make lint`、`make test`、`make integration`、`make check`、
  `git diff --check`。
- 完成报告必须列出实际改动、冻结语义、精确测试结果、未校准项和跨包兼容状态。
