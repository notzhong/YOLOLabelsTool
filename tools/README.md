# tools — 开发复验脚本

本目录收录**只读复验**（本目录根）与**一次性重构**（`mixin_split/`）脚本：把 v2.4.0
重构/修复的关键不变量固化成可重复执行的检查，替代人工点测。

## 环境

```bash
conda activate yolo
export QT_QPA_PLATFORM=offscreen   # 无头模式
```

脚本面向 Linux/macOS 开发机（临时文件写入 `/tmp`）；Windows 下需自行改用 `tempfile`。

## 只读复验脚本（不修改仓库内容）

| 脚本 | 作用 | 关键断言 |
|---|---|---|
| `verify_split.py` | Mixin 拆分保真：把 `src/ui/*_mixins/` 每个方法体与**拆分前原文**（git 基线 `f3d4a66^` / `ffcc902`）做 AST 源码片段逐字节比对 | 非白名单方法必须逐字节一致；`INTENTIONAL` 白名单逐条记录有意行为修复 |
| `audit_self_attrs.py` | AST 扫描 `self.*`：找出"只读未写"的数据属性（AttributeError 高危） | 打印疑似清单（方法名会列出，属正常噪声） |
| `verify_flow_fixes.py` | 操作流程回归：B2 路径锚定 / B3 移除图片重定位 / B5 批量标注刷新画布 / B14 关夹清撤销栈 | 跨 CWD 不产生 config/annotations；移除后停留原图；切图不覆盖批量结果；关夹后不可撤销 |
| `verify_decision_fixes.py` | 决策项回归：B6 撤销按图作用域 / B13 训练"重新配置"流程 / runs 目录锚定 | 换图清栈；重新配置后留在配置窗；正常关闭维持原行为；输出目录锚定应用根 |
| `smoke_dialogs.py` | TrainDialog / ValidationDialog offscreen 实例化 + 关键路径 | 45 个配置键往返、恢复/增量互斥、DPI 坐标换算、Unicode 绘制 |
| `smoke_mainwindow.py` | MainWindow 实例化 + `update_ui_texts` 全链路 | 5 个 Mixin 装配、文案刷新链 |
| `smoke_export_deps.py` | 导出依赖"询问后安装" 7 条路径 | 真实检测 / 拒绝中止 / 同意触发 / 冻结复制命令（剪贴板断言）/ InstallWorker 成功与失败（真实 pip 子进程） |

```bash
python tools/verify_split.py
python tools/verify_flow_fixes.py
python tools/verify_decision_fixes.py
python tools/smoke_mainwindow.py
```

## 维护约定

- 改动 Mixin 方法体后**必须**跑 `verify_split.py`：非有意修复时差异应为空；确需改行为，
  把方法名加入 `INTENTIONAL` 白名单并写明原因（提交信息同样要写）——这正是防止
  "机械搬迁导致相对导入断链"类回归的机制；
- `verify_split.py` 依赖完整 git 历史（基线 ref `f3d4a66^`、`ffcc902`），浅克隆会给出提示；
- 这些是 UI 层脚本，**不进 CI**（`.github/workflows/ci.yml` 不安装 PySide6）；CI 只跑
  `tests/`（core/utils，196 个用例）；
- 修复"用户操作流程"缺陷时，优先把复现逻辑固化为 `verify_*.py` 断言，或下沉为
  `tests/` 下的纯逻辑用例。

## `mixin_split/`：一次性重构脚本（⚠️ 会重写源码）

| 脚本 | 用途（v2.4.0 已执行完毕，仅供复现） |
|---|---|
| `split_train_dialog.py` | 从 `ffcc902:src/ui/train_dialog.py` 机械抽取 4 个 Mixin |
| `split_validation_dialog.py` | 从 `ffcc902:src/ui/validation_dialog.py` 抽取 3 个 Mixin + `unicode_text.py` |
| `fix_main_window_mixins.py` | 用 `f3d4a66^:src/ui/main_window.py` 修正首次拆分的重复空行 |
