# 23 · 训练机任务单（2026-11-10 / v0.28.0）

> 本单只写**训练机上要跑什么、跑完怎么算通过**。
> 测量口径与不变量见 `AGENTS.md`；本次代码变更见 git log。

## 0. 先决条件

```bash
# 训练机侧（Windows 一键）
python tools\sync_agent.py --auto-update      # 拉到 v0.28.0 并覆盖代码
python -m qcore.kit_doctor                    # 自检：依赖 / 数据 / 模型 / 权限
```

服务端（NAS）需要先发布 v0.28.0 的 kit —— 见 §5「发布链路」。
**版本没发布，训练机会一直 skip**（`kit_update` 只在服务端版本更新时才覆盖）。

## 1. 必跑：模型按周期重训（h=1 / h=5 / h=20）

**为什么**：本版把模型与权重器的产物名改成带实例标识 ——
`gbdt_h{h}.pkl` / `meta_h{h}.json` / `current_h{h}.json` / `weighter_{mkt}_h{h}.npz`。
旧代码无论训哪个周期都覆盖 `gbdt.pkl`（陷阱 13）。现在文件名与 `cfg.horizon` 已对齐，
**但存量模型文件还是旧命名，需要重训一次把血缘建立起来**。

```bash
python -m tools.train_one_click --horizon 1
python -m tools.train_one_click --horizon 5
python -m tools.train_one_click --horizon 20
```

**验收**（三条都要满足，缺一条算没通过）：

| 项 | 判据 |
|---|---|
| 产物齐全 | `data/model/gbdt_h{1,5,20}.pkl` + `meta_h{h}.json` + `current_h{h}.json` 都存在 |
| 血缘一致 | `meta_h{h}.json.train_end` 与 `data/state/model.json` 里同周期的 `train_end` **相等** |
| 不互相覆盖 | 训完 h=5 后 `gbdt_h1.pkl` 的 mtime **不变** |

**带回**：三个 `meta_h{h}.json` 的 `ic / ic_t / rank_ic / train_end / n_trees`。

## 2. 必跑：Alpha158 线补四件套

**为什么**：Alpha158+GBDT 声称 IC +0.0260(t=+7.89)，但
① 从不调用 `stats.deflated_sharpe`；② `scripts/robustness.py` 三池用了**不同参数**
（csi300 hold=20，csi500/csi1000 hold=10）。按项目规则这属于**不可采信**。

```bash
python -m scripts.benchmark_alpha158          # 现有基线（会跑截断不变性自检）
python -m scripts.robustness                  # 三池稳健性（当前参数不一致）
```

**要求改动**：把三池统一成同一组 `(k, hold, band)` 重跑一次；`robust_report.json` 每行补
`hac_t`（`lags=h-1`）与 `dsr`（`n_trials` 取真实试验数，不是 73）。

**验收**：`robust_report.json` 每行含 `hac_t` 与 `dsr`；三池参数一致；
结论段不再出现「保守估计 3-6%」这类无统计依据的区间。

## 3. 必跑：`combo.py` 的行为分复算（决定「接入还是作废」）

**为什么**：`qcore/combo.py` 文档串写着「行为分 IC=+0.0352 t=+5.14」——全库最高 IC，
但该模块**零 importer、未登记 claims**，且本机 160 只池实测最大 |IC| 仅 0.0212、
中性化后多数 <0.008。这条线索不能悬着。

```bash
python -m qcore.behavior_report --top-n 283 --start 2013-01-01
```

**验收**（四件套齐全才算数）：

| 项 | 判据 |
|---|---|
| 样本量 | 报出 `n_days` 与 `n_eff = n_days/h` |
| HAC t | `lags=h-1`，不是朴素 t |
| Deflated Sharpe | `n_trials` 取真实试验数并写明构成 |
| 中性化 | 行业 + **历史市值**（不是今日市值）双中性化后的 IC |

**结论只有两种**：`|IC| ≥ 0.03 且 dsr 显著` → 接入并登记 claim；否则**正式作废**并删掉文档串里的数字。

## 4. 可选（有收益但有前置）：MA 族数值稳定化

**为什么**：本版试过「信号前只算最后 N 根」把增量从 16.3s 砍到 ~11s，
但即使 warmup 给到 620 根，仍有 **15/513k 单元格**与全量计算不同 ——
MA 族用前缀和（cumsum）算移动平均，序列变短后浮点舍入改变、在阈值边界翻转 DIR。
这违反 INV-1 的「逐位相等」，所以**优化已回滚**（见 AGENTS §1）。

**要做的事**：把 `qcore/signals.py` 里 MA 族改用数值稳定的 `rolling().mean()`
（或对前缀和做补偿求和），然后验证：

```bash
python -m tests.test_signals          # 19 passed，截断一致性 max|diff|=0.0e+00
# 再验证：截断版 vs 全量版 DIR/STR 逐位相等（临时脚本，不要求入库）
```

**注意**：改 MA 会让**所有历史信号值发生浮点级变化** → 必须重跑
`python -m qcore.pipeline --pool 300 --start 2013-01-01` 并对比新旧
Rank IC / HAC t / DSR，确认没有实质变化才能留。

## 5. 发布链路（本机做不到的部分，需人工）

本机只能推到 GitHub；**训练机从 NAS 拉 kit**，所以还需要两步：

```bash
# ① 打 tag（触发 GitHub Actions 构建镜像与 fpk）
git tag v0.28.0 && git push origin v0.28.0

# ② 构建训练 kit 并发布到 NAS（需要 NAS 访问权限）
bash scripts/build-train-kit.sh           # 产出 apb-train-kit_0.28.0.zip
#   然后把 zip + manifest 放到 NAS 的发布目录（/api/train/kit/download 会用到）
```

`qcore/kit_update.check_and_update()` 比对的是**服务端 manifest 的 version**；
服务端没更新，训练机 `--auto-update` 会打印「已是最新」并跳过。

## 6. 回归门禁（训练机改完代码必须过）

```bash
python tests/test_no_lookahead.py         # worst diff 必须 0.000e+00
python -m tests.test_signals              # 19 passed，截断一致性逐位相等
python audit/leak_scan.py                 # HIGH=0
python -m qcore.invariants                # 8 checks, 0 failed
python -m qcore.claims settle             # 到期判断必须结算
python -m audit.gen_waivers --check       # 豁免清单与代码一致
```

任何一条红了，**先修再训**——训练出来的数字如果建在有已知缺陷的证据源上，等于白跑。

## 7. 交付物清单（训完回传）

| 文件 | 用途 |
|---|---|
| `data/model/meta_h{1,5,20}.json` | 模型血缘（train_end / ic / ic_t） |
| `data/state/performance.json` | 基线 A 全套指标（重训后必须重跑 pipeline 才有） |
| `data/state/basis.json` / `basis_ic.json` | 复权口径探针（本版新增） |
| `robust_report.json` | 三池稳健性（补 hac_t + dsr 后） |
| `data/claims/claims.jsonl` | 新登记的判断（含 falsifier 与 review_by） |
