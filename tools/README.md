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
| `verify_no_image_annotation.py` | 未加载图片即标注的回归：A 冷启动空画布真实派发鼠标按下/移动/松开；B 加载后关夹再拖拽（专测 `_image_item` 残留）；C 无图直调 `add_annotation_with_command` + 图片解码失败回滚 | 三条路径都不抛异常且不产生标注；正常图片拖拽仍可用（守卫未误伤）；关夹后 `canvas.has_image` 为 False |
| `verify_polygon_flow.py` | 分割标注（2.5.0）全流程：A 初始化 / B 绘制+双击闭合 / C Backspace+Esc / D 检测框绘制回归 / E 顶点拖动（手势信号、恰好一步撤销、画布-缓存对象一致性、undo/redo）/ F 绘制中 Ctrl+Z 拦截 / G 导入当前图片 / H 当前图片转换 / I 导出（临时转换确认 + `task: segment`）/ J 任务模式持久化 / K 批量导入匹配 / L 文件夹批量转换 | 12 个场景全部通过；多边形进管理器与撤销栈；一次手势恰一步撤销；导出标签为分割格式 |

```bash
python tools/verify_split.py
python tools/verify_flow_fixes.py
python tools/verify_decision_fixes.py
python tools/verify_no_image_annotation.py
python tools/verify_polygon_flow.py
python tools/smoke_mainwindow.py
```

> `verify_no_image_annotation.py` 自带隔离：用 `tempfile.mkdtemp()` 建图片与标注目录，
> 把 `config_file_path` 指到临时文件，并把 `QMessageBox.*` 换成记账空实现（offscreen 下
> 模态框无人点击会挂死）。不要改回固定路径 `/tmp/xxx`，也不要省掉 config 重定向：
> 标注目录锚定应用根，残留的标注文件会在下次运行时被重新加载，旧标注落在光标下，
> 导致"正常拖拽"断言假失败；而关夹/切图会 `save_settings()` 回写语言与主题，
> 直接跑会改掉你本机 `config/config.ini`。

> `verify_polygon_flow.py` 的鼠标交互必须走 `QTest`（`QTest.mouseClick/mousePress/mouseMove/
> mouseRelease`，坐标用 `canvas.mapFromScene` 换算）：图形项事件（多边形顶点拖动）只有在
> 事件从窗口系统管线进入时才会被视图→场景分发给 item；手搓 `QMouseEvent` 直接调
> `canvas.mousePressEvent` 时二次分发不生效——画布自身逻辑照常，但 item 静默收不到事件。
> 键盘事件（Esc/Backspace）无此问题，可直接构造 `QKeyEvent` 调用 `keyPressEvent`。

## 维护约定

- 改动 Mixin 方法体后**必须**跑 `verify_split.py`：非有意修复时差异应为空；确需改行为，
  把方法名加入 `INTENTIONAL` 白名单并写明原因（提交信息同样要写）——这正是防止
  "机械搬迁导致相对导入断链"类回归的机制；
- `verify_split.py` 依赖完整 git 历史（基线 ref `f3d4a66^`、`ffcc902`），浅克隆会给出提示；
- 这些是 UI 层脚本，**不进 CI**（`.github/workflows/ci.yml` 不安装 PySide6）；CI 只跑
  `tests/`（core/utils，223 个用例）；
- 修复"用户操作流程"缺陷时，优先把复现逻辑固化为 `verify_*.py` 断言，或下沉为
  `tests/` 下的纯逻辑用例。

## `mixin_split/`：一次性重构脚本（⚠️ 会重写源码）

| 脚本 | 用途（v2.4.0 已执行完毕，仅供复现） |
|---|---|
| `split_train_dialog.py` | 从 `ffcc902:src/ui/train_dialog.py` 机械抽取 4 个 Mixin |
| `split_validation_dialog.py` | 从 `ffcc902:src/ui/validation_dialog.py` 抽取 3 个 Mixin + `unicode_text.py` |
| `fix_main_window_mixins.py` | 用 `f3d4a66^:src/ui/main_window.py` 修正首次拆分的重复空行 |
