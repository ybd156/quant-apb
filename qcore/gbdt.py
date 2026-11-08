"""
GBDT —— 纯 numpy 实现的梯度提升树，面向 A 股横截面排序。

## 为什么自己写

微软 Qlib 用 LightGBM + Alpha158 在 CSI300 上做到 IC=0.0448。
但 LightGBM 是 C++ 编译产物，N100 小主机上装它要拉一整套轮子；
本项目契约是「只有 numpy + pandas」。所以这里用 numpy 重写 GBDT 的核心 90%：
直方图分裂 + 二阶梯度 + 叶子正则，这三样决定了 LightGBM 的速度和精度，
剩下的工程细节（缺失值默认方向、类别特征、并行、GPU）在本项目的场景里都不必要。

**不做的事**：不做分位数近似之外的加速（无 OpenMP、无 SIMD 手写），
不做类别特征、不做单调约束、不做缺失值自动分裂方向（NaN 单独占一个箱）。

## 算法骨架（XGBoost 口径）

第 m 棵树拟合当前残差的负梯度：

    g_i = ∂L/∂ŷ_i ,  h_i = ∂²L/∂ŷ_i²           # L2 损失：g = ŷ - y, h = 1

对每个候选分裂，增益（增益越大越该切）：

    Gain = ½[ GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ) ]

叶子取值（把该叶子内的样本推到损失最小）：

    w* = −G / (H + λ) = −Σg / (Σh + λ)

最终预测：

    ŷ_i = Σ_m η · w*_{m, leaf(i)}                # η = learning_rate

注意 Gain 公式里的 ½ 和叶子值里的负号是配套的：Gain 就是「按 w* 切完之后
损失下降了多少」。所以 Gain 可以直接当作特征重要性累加，不需要额外换算。

## 速度来自哪里（这是本文件的核心）

朴素做法是「排序 + 逐点扫描」找阈值：每个节点 O(n·log n) 每特征，20 万行 ×158 特征
根本跑不完。LightGBM 的做法是**先分箱，再在直方图上找阈值**：

1. **分箱**（fit 时做一次）：每个特征用分位点切成 ≤max_bins 个箱，X 变成 uint8。
   此后所有比较都是「箱号比较」，不再碰浮点。
2. **直方图**：一个节点的分裂只统计「每个箱里 g 的和、h 的和、样本数」，
   用 `np.bincount` 一次散射完成，O(n + bins) 而不是 O(n log n)。
3. **前缀和找阈值**：对 64 个箱做 `cumsum`，一次向量化算出全部 63 个候选阈值的增益，
   取 argmax。于是「找最优分裂」= 2~3 次 bincount + 1 次 cumsum。

实测（本机，200k×158）：bincount 散射吞吐 ≈ 440M 元素/秒（float64 权重），
这是整个训练时间的地基。所以这里刻意用 **float64 存 g/h**：numpy 的
`bincount(weights=float32)` 内部会先 astype 成 float64，反而比直接给 float64 慢 60%。

4. **一个免费的优化**：L2 损失下 h≡1，于是「每箱样本数」= 「每箱 h 的和」，
   省掉整整一次 bincount（约 1/3 的分裂耗时）。代码里用 `_counts_from_hess` 标记，
   hess 不恒定时自动回退到独立统计，所以换成别的目标函数也不会错。

5. **直方图相减**（LightGBM 的第二个核心优化）：父子直方图满足
   hist(父) = hist(左) + hist(右)，所以只从原始数据统计**较小**的子节点，
   另一个用 `父 − 小子` 算出来。每层省掉一半散射量：
   深度 4 的总代价从 4n 降到 n + 3×(n/2) = 2.5n。实测 68.5s → 44.4s（1.54×）。

   这里有个**必须记住的坑**：只有「样本数」直方图可以钳到非负，
   梯度直方图 hist_g **绝不能**钳 —— g = ŷ − y 天然是负的，钳了等于把
   右子树的梯度全抹平。这个 bug 曾让验证集 R² 从 0.73 掉到 0.50。

   同理，节点级 Σg/Σh 要用 `g[idx].sum()` 精确算，不能用直方图求和：
   相减噪声虽然只有 1e-12，但 GR = G − GL 是逐箱共用的，会被系统性放大，
   实测让 R² 从 0.73 掉到 0.70（5 个随机种子一致复现）。

## 实测（本机，非 N100）

    合成数据 50k×20，R² 0.7257（理论上限 0.7429，达到 97.7%）
    规模压测 200k×158，100 棵树，深度 4：44.4 s（444 ms/树）

## 数据布局

分箱后的数据存成 **(K, n) 列主序 uint8**（不是 (n, K)）。
因为分裂时永远是「取某个特征的一列」，列主序下 `Xb[k]` 是连续内存，
`Xb[k].take(idx)` 一次 gather 就得到该节点该特征的全部箱号；
行主序则每取一列要跨 158 个元素跳一次，实测慢 2 倍以上。
200k×158 的 uint8 面板 = 31.6MB，缓存友好。

## 用法

    from qcore.gbdt import GBDT
    m = GBDT(n_estimators=100, max_depth=4).fit(Xtr, ytr, eval_set=(Xva, yva))
    pred = m.predict(Xte)          # 横截面排序直接用 pred 的秩
    imp = m.feature_importance     # 归一化增益，和为 1
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["GBDT"]


class GBDT:
    """梯度提升回归树（直方图版）。

    参数
    ----
    n_estimators : 树的数量（上限，早停会截断）
    max_depth : 最大深度。4 层 = 16 个叶子，对 A 股因子够用；
                再深会开始记忆噪声（本项目横截面 IC 只有 0.04 量级，树深了必过拟合）
    learning_rate : 学习率 η，叶子输出乘它再累加
    min_samples_leaf : 叶子最少样本数（按行采样后的样本计）。这是抗过拟合的第一道闸，
                横截面数据里它比 max_depth 更重要
    subsample : 行采样比例（每棵树重抽，Friedman 的 stochastic boosting）
    colsample : 列采样比例（每棵树重抽，LightGBM 的 feature_fraction）
    l2 : 叶子值的 L2 正则 λ，出现在 w* = -G/(H+λ) 和增益公式的分母里
    max_bins : 每个特征的最大分箱数（含 1 个 NaN 专用箱），≤255（uint8 上限）
    random_state : 随机种子，固定后 fit 完全可复现
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 100,
        subsample: float = 0.8,
        colsample: float = 0.8,
        l2: float = 1.0,
        max_bins: int = 64,
        random_state: int = 42,
    ) -> None:
        if not (1 <= int(max_bins) <= 255):
            raise ValueError("max_bins 必须在 [1, 255]（分箱用 uint8 存储）")
        if int(max_depth) < 0:
            raise ValueError("max_depth 必须 >= 0")
        if int(min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf 必须 >= 1")
        if not (0.0 < float(subsample) <= 1.0):
            raise ValueError("subsample 必须在 (0, 1]")
        if not (0.0 < float(colsample) <= 1.0):
            raise ValueError("colsample 必须在 (0, 1]")
        if float(l2) < 0.0:
            raise ValueError("l2 必须 >= 0")
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate 必须 > 0")

        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.l2 = float(l2)
        self.max_bins = int(max_bins)
        self.random_state = int(random_state)

        # 以下由 fit 填充
        self._edges: list[np.ndarray] = []
        self._nbins: int = self.max_bins
        self._n_features: int = 0
        self._trees: list[dict] = []
        self._gain: np.ndarray | None = None
        self._counts_from_hess: bool = True

    # ------------------------------------------------------------------ #
    # 分箱：fit 时做一次，把浮点特征压成 uint8 箱号
    # ------------------------------------------------------------------ #
    def _build_bins(self, X: np.ndarray) -> None:
        """为每个特征算分位点切分边界。

        为什么用分位点而不是等宽：因子分布普遍尖峰厚尾（换手率、振幅），
        等宽分箱会把 90% 的样本塞进一两个箱，直方图失去分辨率。
        分位点保证每个箱样本量大致相等，阈值搜索的分辨率在尾部也不会退化。

        抽样：分位点用最多 10 万行估计（确定性步长抽样，不消耗随机数，
        所以不影响可复现性）。20 万行的分位数和 10 万行的分位数差在小数点后 4 位，
        对「箱边界」这种粗粒度用途没有影响，但省掉一半排序时间。
        """
        n, K = X.shape
        nb = self._nbins
        n_edges = nb - 1  # 真实取值用 0..nb-2 箱，第 nb-1 箱留给 NaN
        edges: list[np.ndarray] = []
        for k in range(K):
            col = X[:, k]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                # 全 NaN 的特征：没有边界，全部落进 NaN 箱，永远分不出东西
                edges.append(np.empty(0, dtype=np.float32))
                continue
            step = max(1, finite.size // 100_000)
            sub = finite[::step]
            # 去掉 0% 和 100% 两个端点（端点会把极端值单独关进一个箱）
            qs = np.linspace(0.0, 1.0, n_edges + 2)[1:-1]
            e = np.unique(np.quantile(sub, qs)).astype(np.float32)
            edges.append(e)
        self._edges = edges

    def _bin(self, X: np.ndarray) -> np.ndarray:
        """把 (n,K) 浮点矩阵映射成 (K,n) uint8 箱号矩阵（列主序）。"""
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X 必须是二维 (n, K)")
        n, K = X.shape
        if self._n_features and K != self._n_features:
            raise ValueError(f"特征数不匹配：fit 时 {self._n_features}，现在 {K}")
        nb = self._nbins
        out = np.empty((K, n), dtype=np.uint8)
        for k in range(K):
            col = X[:, k]
            e = self._edges[k]
            if e.size == 0:
                b = np.zeros(n, dtype=np.uint8)
            else:
                # searchsorted(side='right')：b = 「≤x 的边界个数」，
                # 于是 b 就是 x 所属的箱号，b ∈ [0, len(e)]
                b = np.searchsorted(e, col).astype(np.uint8)
                np.clip(b, 0, e.size - 1, out=b)  # 顶端并入最后一箱
            if np.isnan(col).any():
                b[np.isnan(col)] = nb - 1  # NaN 独占最高箱
            out[k] = b
        return out

    # ------------------------------------------------------------------ #
    # 核心：在一个节点上找最优分裂
    # ------------------------------------------------------------------ #
    def _node_hist(self, Xb, cols, idx, gnode, hnode, nb):
        """统计一个节点的直方图，返回 (hist_g, hist_h, hist_c)。

        形状都是 (ng, nbins)：第 j 行是第 cols[j] 个特征的 64 个箱。
        一次 bincount 就是「把每行的 g 散射加到它所属的箱上」，O(n + bins)。
        这就是 LightGBM 快的根：不排序、不逐点扫描，纯散射加法。

        hist_c（每箱样本数）在 hess 恒定时返回 None —— 那时它等于 hist_h / h，
        由 `_split_from_hist` 直接换算，省掉整整一次 bincount。
        """
        ng = cols.size
        hist_g = np.empty((ng, nb))
        hist_h = np.empty((ng, nb))
        need_count = not self._counts_from_hess
        hist_c = np.empty((ng, nb)) if need_count else None
        for j in range(ng):
            b = Xb[cols[j]].take(idx)
            hist_g[j] = np.bincount(b, weights=gnode, minlength=nb)
            hist_h[j] = np.bincount(b, weights=hnode, minlength=nb)
            if need_count:
                hist_c[j] = np.bincount(b, minlength=nb)
        return hist_g, hist_h, hist_c

    def _split_from_hist(self, hist, cols, nb, n_node, G, H):
        """在直方图上找最优分裂，返回 (特征, 箱阈值, 增益) 或 None。

        两步，都不再碰原始数据：
        ① cumsum 把「每箱统计量」变成「左子集统计量」，一次算出全部候选阈值
        ② 增益公式向量化取 argmax

        G/H 由调用方传入**精确**的节点梯度和/海森和（而不是直方图求和）：
        直方图相减会带来 1e-12 量级的浮点噪声，而 GR = G − GL 是逐箱共用的，
        这点噪声会系统性地扭曲近似平局时的 argmax，实测让验证集 R² 从 0.73 掉到 0.70。
        """
        hist_g, hist_h, hist_c = hist
        l2 = self.l2

        # ① 前缀和：切在第 t 箱之后（左 = 箱 0..t），所以有 nb-1 个候选阈值
        GL = np.cumsum(hist_g, axis=1)[:, :-1]  # 左子集 Σg
        HL = np.cumsum(hist_h, axis=1)[:, :-1]  # 左子集 Σh
        GR = G - GL
        HR = H - HL

        if hist_c is None:
            # h 恒定 ⇒ 每箱样本数 = 每箱 h 之和 × n / H，白捡一个直方图
            scale = n_node / H if H > 0 else 0.0
            CL = HL * scale
            CR = HR * scale
        else:
            CL = np.cumsum(hist_c, axis=1)[:, :-1]
            CR = n_node - CL

        # ② 增益：Gain = ½[GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ)]
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = 0.5 * (
                GL * GL / (HL + l2) + GR * GR / (HR + l2) - G * G / (H + l2)
            )
        # 合法性：两边都得够 min_samples_leaf，且 h 不为 0（否则该侧其实是空的）。
        # 叶子数比较留 1e-6 容差：直方图相减会带来 1e-12 量级的浮点噪声，
        # 恰好卡在 min_samples_leaf 上的箱子不该因为这个噪声被判非法。
        tol = 1e-6
        msl = self.min_samples_leaf
        ok = (CL >= msl - tol) & (CR >= msl - tol)
        ok &= (HL > 0.0) & (HR > 0.0)
        gain = np.where(ok, gain, -np.inf)

        flat = int(np.argmax(gain))
        best = float(gain.flat[flat])
        if not np.isfinite(best) or best <= 0.0:
            return None
        j, t = divmod(flat, nb - 1)
        return int(cols[j]), int(t), best

    def _leaf_value(self, G: float, H: float) -> float:
        """叶子值 w* = −Σg/(Σh+λ)，把叶子内的样本推到损失最小。"""
        return -G / (H + self.l2)

    # ------------------------------------------------------------------ #
    # 建一棵树
    # ------------------------------------------------------------------ #
    def _build_tree(self, Xb, g, h, rows, cols, nb):
        """深度优先建树，节点用扁平数组存（预测时才好向量化）。

        **直方图相减**（LightGBM 的第二个核心优化）：
        一个节点分裂后，两个子节点的直方图满足
            hist(父) = hist(左) + hist(右)
        所以只从原始数据统计**较小**的那个子节点，另一个直接 `父 − 小子`。
        每层省掉一半的散射量：深度 4 时总代价从 4n 降到 n + 3×(n/2) = 2.5n。

        叶子节点把 left/right 指向自己（自环），这样预测循环里不需要判断
        「这一行已经到叶子了」，直接迭代 max_depth+1 次就够，代码短且快。
        """
        feat: list[int] = [0]
        sbin: list[int] = [0]
        left: list[int] = [-1]
        right: list[int] = [-1]
        val: list[float] = [0.0]
        leaf: list[bool] = [True]
        gain_acc = np.zeros(self._n_features)

        # 根节点直方图：全树唯一一次「从原始数据全量统计」
        root_hist = self._node_hist(Xb, cols, rows, g[rows], h[rows], nb)
        stack = [(0, rows, 0, root_hist)]
        while stack:
            nid, idx, depth, hist = stack.pop()
            # 节点级 Σg / Σh：一次 gather 直接求和，保证后续增益与叶子值都用精确值
            G = float(g[idx].sum())
            H = float(h[idx].sum())
            # 提前判叶：深度到顶，或样本数不够切出两个合法叶子（省掉整趟直方图）
            if depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf:
                val[nid] = self._leaf_value(G, H)
                continue
            res = self._split_from_hist(hist, cols, nb, idx.size, G, H)
            if res is None:
                val[nid] = self._leaf_value(G, H)
                continue
            k, t, gn = res
            # 按箱号切分：左 = 箱 ≤ t。uint8 比较，比浮点比较省一次类型转换
            b = Xb[k].take(idx)
            mask = b <= t
            lidx = idx[mask]
            ridx = idx[~mask]
            if lidx.size == 0 or ridx.size == 0:  # 理论上不会发生，兜底
                val[nid] = self._leaf_value(G, H)
                continue

            # ---- 直方图相减：只统计较小的子节点 ----
            hg_p, hh_p, hc_p = hist
            if lidx.size <= ridx.size:
                s_idx = lidx
            else:
                s_idx = ridx
            s_g, s_h = g[s_idx], h[s_idx]
            s_hg, s_hh, s_hc = self._node_hist(Xb, cols, s_idx, s_g, s_h, nb)
            o_hg = hg_p - s_hg
            o_hh = hh_p - s_hh
            # 注意：**绝不能**把 hist_g 钳到非负。梯度之和天然是负的
            # （g = ŷ − y），钳了就等于把右半边的梯度全抹平。
            # 只有「样本数」必须非负，其余交给后面的合法性检查兜底。
            np.maximum(o_hh, 0.0, out=o_hh)  # 海森恒非负，钳掉相减噪声
            if hc_p is not None:
                o_hc = hc_p - s_hc
                np.maximum(o_hc, 0.0, out=o_hc)
            else:
                o_hc = None
            if lidx.size <= ridx.size:  # 小的那半是左孩子
                lhist = (s_hg, s_hh, s_hc)
                rhist = (o_hg, o_hh, o_hc)
            else:
                rhist = (s_hg, s_hh, s_hc)
                lhist = (o_hg, o_hh, o_hc)

            lc = len(feat)
            for _ in range(2):  # 预占左右子节点
                feat.append(0)
                sbin.append(0)
                left.append(-1)
                right.append(-1)
                val.append(0.0)
                leaf.append(True)
            rc = lc + 1
            feat[nid] = k
            sbin[nid] = t
            left[nid] = lc
            right[nid] = rc
            leaf[nid] = False
            gain_acc[k] += gn
            stack.append((lc, lidx, depth + 1, lhist))
            stack.append((rc, ridx, depth + 1, rhist))

        for i in range(len(feat)):
            if leaf[i]:  # 自环
                left[i] = i
                right[i] = i

        tree = {
            "feat": np.asarray(feat, dtype=np.int32),
            "sbin": np.asarray(sbin, dtype=np.int32),
            "left": np.asarray(left, dtype=np.int32),
            "right": np.asarray(right, dtype=np.int32),
            "value": np.asarray(val, dtype=np.float64),
            "depth": self.max_depth,
        }
        return tree, gain_acc

    # ------------------------------------------------------------------ #
    # 预测
    # ------------------------------------------------------------------ #
    @staticmethod
    def _predict_tree(tree: dict, Xb: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """对指定行向量化下树。全部行一起走，每层一次 gather。"""
        feat = tree["feat"]
        sbin = tree["sbin"]
        left = tree["left"]
        right = tree["right"]
        value = tree["value"]
        node = np.zeros(rows.size, dtype=np.int32)
        for _ in range(tree["depth"] + 1):
            b = Xb[feat[node], rows]  # 每行取「它当前节点用到的那个特征」的箱号
            node = np.where(b > sbin[node], right[node], left[node])
        return value[node]

    def _predict_binned(self, Xb: np.ndarray) -> np.ndarray:
        rows = np.arange(Xb.shape[1])
        out = np.zeros(Xb.shape[1], dtype=np.float64)
        for tree in self._trees:
            out += self._predict_tree(tree, Xb, rows)
        return out * self.learning_rate  # 所有树 × η 累加

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测。输入与 fit 同列；返回 (n,) float64。"""
        if not self._trees:
            raise RuntimeError("模型未训练，先调用 fit")
        return self._predict_binned(self._bin(X))

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_eval_set(eval_set):
        """接受 (X,y) 或 [(X,y), ...]，取最后一个作早停集。"""
        if isinstance(eval_set, tuple):
            sets = [eval_set]
        else:
            sets = list(eval_set)
        Xv, yv = sets[-1]
        return (
            np.ascontiguousarray(Xv, dtype=np.float32),
            np.asarray(yv, dtype=np.float64).ravel(),
        )

    def fit(self, X, y, eval_set=None, early_stopping_rounds: int = 20):
        """训练。返回 self。

        eval_set 给了就在验证集上算 L2，连续 early_stopping_rounds 轮不改善则
        截断到最优轮（不是「停手但保留全部树」—— 保留会白算，还让验证集精度倒退）。
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X 必须是二维 (n, K)")
        y = np.asarray(y, dtype=np.float64).ravel()
        n, K = X.shape
        if y.size != n:
            raise ValueError(f"X/y 行数不一致：{n} vs {y.size}")
        if n < 2:
            raise ValueError("样本数太少")

        self._n_features = K
        self._nbins = self.max_bins
        self._build_bins(X)          # ① 一次性分箱
        Xb = self._bin(X)            # ② 全量压成 uint8 面板
        rng = np.random.default_rng(self.random_state)
        nb = self._nbins

        pred = np.zeros(n, dtype=np.float64)
        trees: list[dict] = []
        gain_total = np.zeros(K, dtype=np.float64)
        self.evals_result_: dict[str, list[float]] = {"train_l2": [], "valid_l2": []}

        have_eval = eval_set is not None
        if have_eval:
            Xv, yv = self._split_eval_set(eval_set)
            Xvb = self._bin(Xv)
            vrows = np.arange(yv.size)
            predv = np.zeros(yv.size, dtype=np.float64)
            best_l2 = np.inf
            best_iter = -1
            no_improve = 0

        for it in range(self.n_estimators):
            # ---- 行采样：每棵树用不同的子集，降低树间相关性 ----
            if self.subsample < 1.0:
                rows = np.flatnonzero(rng.random(n) < self.subsample)
                if rows.size < 2 * self.min_samples_leaf:
                    rows = np.arange(n)  # 采样太狠就退回全量，别把树饿死
            else:
                rows = np.arange(n)

            # ---- 列采样：每棵树只看一部分特征 ----
            if self.colsample < 1.0:
                kk = max(1, int(round(self.colsample * K)))
                cols = np.sort(rng.choice(K, size=kk, replace=False))
            else:
                cols = np.arange(K)

            # ---- 二阶梯度（L2：g = ŷ − y, h = 1）----
            g = pred - y
            h = np.ones(n, dtype=np.float64)
            # hess 恒定时「每箱样本数」可由「每箱 h 之和」直接得出，省一次 bincount。
            # 换损失（如 Huber）后 h 不再恒定，这里会自动置 False 走独立统计。
            self._counts_from_hess = bool(h.min() == h.max() and h[0] > 0.0)

            tree, gain = self._build_tree(Xb, g, h, rows, cols, nb)
            trees.append(tree)
            gain_total += gain
            pred[rows] += self.learning_rate * self._predict_tree(tree, Xb, rows)

            self.evals_result_["train_l2"].append(float(np.mean((pred - y) ** 2)))

            if have_eval:
                predv += self.learning_rate * self._predict_tree(tree, Xvb, vrows)
                l2v = float(np.mean((predv - yv) ** 2))
                self.evals_result_["valid_l2"].append(l2v)
                if l2v < best_l2 - 1e-12:
                    best_l2 = l2v
                    best_iter = it
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= early_stopping_rounds:
                        trees = trees[: best_iter + 1]  # 回退到最优轮
                        break

        self._trees = trees
        self._gain = gain_total
        self.n_trees_ = len(trees)
        self.best_iteration_ = (
            best_iter if have_eval and best_iter >= 0 else len(trees) - 1
        )
        if have_eval:
            self.best_score_ = best_l2
        return self

    # ------------------------------------------------------------------ #
    @property
    def feature_importance(self) -> np.ndarray:
        """(K,) 归一化到和为 1 的增益重要性。

        用「增益」而不是「分裂次数」：次数会被高频但无信息量的特征刷高
        （例如某个只有 2 个箱的哑变量），增益才反映它真的把损失降了多少。
        """
        if self._gain is None:
            raise RuntimeError("模型未训练，先调用 fit")
        s = float(self._gain.sum())
        if s <= 0.0:
            return np.zeros(self._n_features, dtype=np.float64)
        return self._gain / s

    def __repr__(self) -> str:
        nt = getattr(self, "n_trees_", 0)
        return (
            f"GBDT(n_estimators={self.n_estimators}, trees={nt}, "
            f"max_depth={self.max_depth}, lr={self.learning_rate}, "
            f"max_bins={self.max_bins}, l2={self.l2})"
        )


# ---------------------------------------------------------------------- #
# 自检
# ---------------------------------------------------------------------- #
def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _selfcheck() -> None:
    rng = np.random.default_rng(20261108)

    print("=" * 68)
    print("合成数据自检：y = 2*x0 - 1.5*x1 + 0.5*x0*x2 + noise")
    print("=" * 68)

    n, K = 50_000, 20
    X = rng.standard_normal((n, K)).astype(np.float32)
    y = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.5 * X[:, 0] * X[:, 2]   # 交互项：单棵树要靠多层才能拟合
        + rng.standard_normal(n).astype(np.float32) * 1.5
    ).astype(np.float32)

    ntr = 40_000
    Xtr, ytr = X[:ntr], y[:ntr]
    Xva, yva = X[ntr:], y[ntr:]

    model = GBDT(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=100,
        subsample=0.8,
        colsample=0.8,
        l2=1.0,
        max_bins=64,
        random_state=42,
    )

    t0 = time.perf_counter()
    model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=20)
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    p_tr = model.predict(Xtr)
    p_va = model.predict(Xva)
    pred_sec = time.perf_counter() - t1

    r2_tr = _r2(ytr, p_tr)
    r2_va = _r2(yva, p_va)

    print(f"模型           : {model}")
    print(f"实际树数       : {model.n_trees_}（最优轮 {model.best_iteration_}，上限 200）")
    print(f"训练集 R²      : {r2_tr:.4f}")
    print(f"验证集 R²      : {r2_va:.4f}   <- 要求 > 0.5")
    print(f"训练耗时       : {train_sec:.2f} s（含分箱）")
    print(f"预测耗时       : {pred_sec:.3f} s（{Xva.shape[0]} 行）")

    imp = model.feature_importance
    order = np.argsort(imp)[::-1][:5]
    print("\ntop-5 重要特征（归一化增益）:")
    for rank, j in enumerate(order, 1):
        print(f"  {rank}. x{j:<3d} {imp[j]:.4f}  {'#' * int(imp[j] * 200)}")

    # 可复现性：同输入两次预测必须逐位相同
    a = model.predict(Xva)
    b = model.predict(Xva)
    assert np.array_equal(a, b), "predict 不可复现！"
    print("\n可复现性       : 两次 predict 逐位相同 ✓")

    # 再验一次：同种子重新 fit，预测也必须相同
    m2 = GBDT(
        n_estimators=200, max_depth=4, learning_rate=0.1, min_samples_leaf=100,
        subsample=0.8, colsample=0.8, l2=1.0, max_bins=64, random_state=42,
    ).fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=20)
    assert np.array_equal(m2.predict(Xva), a), "同种子重训结果不一致！"
    print("同种子重训     : 逐位一致 ✓")

    assert r2_va > 0.5, f"验证集 R²={r2_va:.4f} 未达到 0.5"
    top3 = set(order[:3].tolist())
    assert top3 == {0, 1, 2}, f"top-3 特征应为 x0/x1/x2，实际 {sorted(top3)}"
    print("\n全部断言通过 ✓")


def _perf() -> None:
    """规模压测：20 万行 × 158 特征 × 100 棵树 × 深度 4。"""
    rng = np.random.default_rng(7)
    n, K = 200_000, 158
    print("=" * 68)
    print(f"规模压测：{n} 行 × {K} 特征，100 棵树，深度 4")
    print("=" * 68)
    t0 = time.perf_counter()
    X = rng.standard_normal((n, K)).astype(np.float32)
    # 混入真实因子该有的结构：一部分特征带信号，一部分纯噪声
    w = np.zeros(K, dtype=np.float32)
    w[:10] = rng.standard_normal(10)
    y = (X[:, :10] @ w[:10] + 0.3 * X[:, 0] * X[:, 3]
         + rng.standard_normal(n).astype(np.float32) * 2.0).astype(np.float32)
    print(f"造数耗时       : {time.perf_counter() - t0:.2f} s")

    m = GBDT(n_estimators=100, max_depth=4, learning_rate=0.1,
             min_samples_leaf=100, subsample=0.8, colsample=0.8,
             l2=1.0, max_bins=64, random_state=42)
    t0 = time.perf_counter()
    m.fit(X, y)
    dt = time.perf_counter() - t0
    print(f"训练耗时       : {dt:.1f} s  ({dt/100*1000:.0f} ms/树)")
    print(f"要求           : < 180 s -> {'达标 ✓' if dt < 180 else '未达标 ✗'}")
    t0 = time.perf_counter()
    p = m.predict(X[:50_000])
    print(f"预测 5 万行    : {time.perf_counter() - t0:.2f} s，R²={_r2(y[:50_000], p):.4f}")


if __name__ == "__main__":
    import sys

    # Windows 控制台默认 GBK，打不出 R² 这类字符；这里强制 UTF-8 并允许替换
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if "--perf" in sys.argv:
        _perf()
    else:
        _selfcheck()
