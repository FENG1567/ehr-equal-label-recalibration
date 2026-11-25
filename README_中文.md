# 项目复现材料

本项目对应论文 *Residual Differences in In-Hospital Mortality Prediction After Recalibrating Transferred EHR Models With the Same Local Labels*。公开代码版本为 [v1.0.0](https://github.com/FENG1567/ehr-equal-label-recalibration/tree/v1.0.0)，对应投稿材料版本 **20261125-r9**；r9只在r8科研内容基础上加入Code availability声明。当前图表编号以 `data/presentation/figure_display_index.json` 中的 **2026.11.25-r7** 为准。作者：Shuo Feng、Chao Shi、Tianming Liu、Haoran Ding、Fenghe Hang、Zhenghao Huang、Liwei Liu。联系邮箱：18940835066@163.com。

## 快速开始

使用Python 3.12和完整锁定的公开依赖环境：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r envs/requirements-public.lock.txt
python scripts/check_release.py
python scripts/reproduce.py --verify-only
python -m unittest discover -s tests -v
python scripts/reproduce.py --output outputs
```

该流程从汇总CSV生成6幅主图和6幅补充图，并导出PDF、SVG、PNG、TIFF及预览PNG；另从汇总CSV重建2张主表和9张补充物理表，逐格核对 `data/presentation/tables.json`。投稿英文稿中的2张主表按r9文档、补充材料9张表按r8文档逐格核对；r9未改变r8的科研表格内容。比较时统一空白、换行和Unicode减号/破折号形式。外部文件指针清理后，表格注释或标题可能不同；文档哈希和核对结果见 `provenance/submission-document-bindings.json`。主Table 2为8行raw-to-Platt摘要；完整24行校准网格保留在Supplementary Tables 2A–2D（每张6行、6列）。补充表依次为临床特征S1、完整校准网格S2A–S2D、Data22均衡标签摘要S3、时间敏感性S4、SICdb S5和点估计对数损失S6。旧7表规范快照另存为 `tables_source_r2.json`。流程不读取患者记录、不重新拟合模型、不重跑患者聚类bootstrap。数值表格和图表结果来自已发布的汇总数据；不同平台生成的PDF字节和字体可能不同。`reference/figures/`保存12个未改字节的v12 PDF，编号以当前r7索引为准。生成输出由Git忽略。

## 内容

| 目录 | 内容 |
|---|---|
| `data/aggregates/Data1`–`Data22` | 作者汇总结果；按 `docs/PUBLIC_DATA_POLICY.md` 省略低支持度站点行 |
| `data/presentation` | 历史图表选择器、当前r7图表编号索引、修订后的英文表格快照和旧7表规范快照 |
| `src/presentation` | 汇总表重建程序和两个v12图表渲染器 |
| `archive/original` | 保留历史路径及保护措施的原始分析代码 |
| `config` | 变量映射及临床阶段/输入元数据 |
| `docs` | 数据获取、临床流程、限制及数据字典 |
| `provenance` | 源码哈希、结果到代码映射、图表绑定、投稿文档表格绑定和发布输入哈希 |
| `tests` | 汇总数据到表格一致性及发布完整性检查 |

基于临床数据重新拟合统计模型时，参阅 `docs/CLINICAL_REPRODUCTION.md`。这需要合法授权的数据库访问及原分析流水线输入布局。归档脚本不是可移植的一键临床重分析流程。历史checkpoint成员身份无法追溯认证；更新的source-only和按日历分隔的分析有独立来源记录。不能通过重命名文件或重建候选名单消除历史不确定性。

## 许可与可用性

作者编写的软件采用 **MIT** 许可；作者生成的汇总结果和展示材料采用 **CC BY 4.0** 许可。参阅 `LICENSING.md` 与 `THIRD_PARTY_NOTICES.md`；外部数据库和软件继续适用其原许可。

复现图表和表格所需的分析代码、软件依赖及汇总结果均可从[公开v1.0.0版本](https://github.com/FENG1567/ehr-equal-label-recalibration/tree/v1.0.0)获取。需要访问原始临床数据库的分析说明见 `docs/CLINICAL_REPRODUCTION.md`。公开流程复现的是汇总展示结果，不声称完成临床数据全流程重跑。开放站点级数据遵循 `docs/PUBLIC_DATA_POLICY.md`。

## 历史图表编号来源

当前图表编号以 `data/presentation/figure_display_index.json`（版本2026.11.25-r7）为准。该文件及 `provenance/figure-source-bindings.json` 保留的r4重编号表只记录历史展示变化：此前主图Figure1–Figure6对应r4 Figure2、Figure6、Figure1、Figure3、Figure4、Figure5；此前补充图FigureS1–FigureS6对应r4 FigureS3、FigureS5、FigureS4、FigureS2、FigureS6、FigureS1。历史source身份及冻结的 `figures.json` source bundle保持不变。此映射仅作来源记录，不表示当前图表编号。

两份汇总CSV超过GitHub网页上传的25 MiB限制；如需上传替代副本，请使用GitHub Desktop或Git CLI：`Data15/sites/site_adaptation_repeat_scores.csv`（51,447,891字节）及 `Data5/reviewed_NWICU_SICdb/transport_calibration_draws.csv`（45,323,976字节）。二者均低于GitHub普通文件100 MB上限。
