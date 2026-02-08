from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Sequence, Set, Optional, Iterable
import itertools
import math


# -----------------------------
# 数据结构
# -----------------------------
# 一个球的表示：(color_name, idx)  idx ∈ {1,2,3}
Ball = Tuple[str, int]
# 状态：最多2个球，按固定顺序排序后的tuple
State = Tuple[Ball, ...]


@dataclass(frozen=True)
class GameConfig:
    # 基础判定概率列表，例如 [1.0, 0.5, 0.3]  长度固定为3（对应idx=1..3）
    q: Sequence[float]
    # 颜色权重（不要求归一化）
    weights: Dict[str, float]
    # 目标颜色集合（Y<=3，如 {"red","white","blue"}）
    target_colors: Set[str]
    # 跨轮最多保留球数（你规则为2）
    carry_limit: int = 2
    # 价值迭代参数
    vi_max_iter: int = 50_000
    vi_tol: float = 1e-12


# -----------------------------
# 工具：权重归一化与校验
# -----------------------------
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    if not w:
        raise ValueError("weights 不能为空")
    for k, v in w.items():
        if v < 0:
            raise ValueError(f"权重不能为负: {k}={v}")
    s = sum(w.values())
    if s <= 0:
        raise ValueError("权重总和必须 > 0")
    return {k: v / s for k, v in w.items()}


def validate_cfg(cfg: GameConfig) -> GameConfig:
    if len(cfg.q) != 3:
        raise ValueError("本模板假设基础判定列表长度固定为3（idx=1..3）")
    for i, p in enumerate(cfg.q, start=1):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"q{i} 必须在[0,1]，当前{p}")
    w = normalize_weights(dict(cfg.weights))
    missing = cfg.target_colors - set(w.keys())
    if missing:
        raise ValueError(f"target_colors 中存在未知颜色: {missing}")
    if len(cfg.target_colors) == 0:
        raise ValueError("target_colors 不能为空")
    if len(cfg.target_colors) > 3:
        raise ValueError("按你的设定，Y最大为3，这里限制 target_colors <= 3")
    if cfg.carry_limit != 2:
        # 你当前规则固定2；想扩展也行，但要留意状态规模
        raise ValueError("本模板按你的规则固定 carry_limit=2")
    return GameConfig(
        q=tuple(cfg.q),
        weights=w,
        target_colors=set(cfg.target_colors),
        carry_limit=cfg.carry_limit,
        vi_max_iter=cfg.vi_max_iter,
        vi_tol=cfg.vi_tol,
    )


# -----------------------------
# 本轮判定：删掉某些idx后，N的分布
# -----------------------------
def n_distribution_after_deletions(q: Sequence[float], deleted: Set[int]) -> List[float]:
    """
    q长度=3，deleted是要删除的idx集合，例如{1,3}
    返回 P(N=n) for n=0..3（实际只有最多len(remaining)）
    N = sum_{i in remaining} Bern(q_i)
    """
    remaining = [q[i - 1] for i in (1, 2, 3) if i not in deleted]
    max_n = len(remaining)
    dist = [0.0] * (max_n + 1)
    dist[0] = 1.0
    for p in remaining:
        new = [0.0] * (max_n + 1)
        for n in range(max_n + 1):
            if dist[n] == 0:
                continue
            new[n] += dist[n] * (1 - p)
            if n + 1 <= max_n:
                new[n + 1] += dist[n] * p
        dist = new
    return dist  # n=0..max_n


# -----------------------------
# 加权无放回抽样：枚举所有“有序序列”及其概率（n<=3时非常可行）
# -----------------------------
def enumerate_weighted_sequences(weights: Dict[str, float], n: int) -> Dict[Tuple[str, ...], float]:
    """
    从weights（已归一化）的颜色中，按“抽到就移除”规则抽n次，返回所有有序序列及其概率。
    n<=3时，用递归枚举即可。
    """
    if n <= 0:
        return {(): 1.0}
    if n > len(weights):
        # 不可能抽超过颜色数（同色移除）
        return {}

    out: Dict[Tuple[str, ...], float] = {}

    def rec(current_w: Dict[str, float], k: int, prefix: Tuple[str, ...], prob: float):
        if k == 0:
            out[prefix] = out.get(prefix, 0.0) + prob
            return
        total = sum(current_w.values())
        if total <= 0:
            return
        # 当前步的选择概率 = w[c] / total
        for c, w in list(current_w.items()):
            if w <= 0:
                continue
            p = w / total
            next_w = dict(current_w)
            next_w.pop(c, None)  # 抽到即移除
            rec(next_w, k - 1, prefix + (c,), prob * p)

    rec(dict(weights), n, (), 1.0)
    return out


# -----------------------------
# 成功判定：某一轮过程中是否集齐目标
# -----------------------------
def hits_target_during_round(start_colors: Set[str], seq: Tuple[str, ...], target: Set[str]) -> bool:
    owned = set(start_colors)
    if target.issubset(owned):
        return True
    for c in seq:
        owned.add(c)
        if target.issubset(owned):
            return True
    return False


# -----------------------------
# 状态空间枚举（最多2个球，每个球idx∈{1,2,3}）
# -----------------------------
def canonical_state(balls: Iterable[Ball]) -> State:
    # 排序保证状态唯一
    return tuple(sorted(balls, key=lambda x: (x[0], x[1])))


def all_states(colors: List[str]) -> List[State]:
    # 空状态
    balls: List[Ball] = []
    all_balls = [(c, i) for c in colors for i in (1, 2, 3)]
    states = [()]  # type: ignore
    # 1个球
    for b in all_balls:
        states.append((b,))
    # 2个球（不允许同一个(ball)重复；允许同色不同idx共存吗？——按你的规则可以（比如不同轮得到同色？）
    # 但你还规定“持有的颜色会从袋子剔除”，因此同色持有两份没有意义且会产生歧义。
    # 我这里默认：允许同色不同idx作为“两个球”，但会导致袋子剔除同色一次即可。
    # 若你希望禁止“同色双持”，可以在这里加过滤。
    for b1, b2 in itertools.combinations(all_balls, 2):
        states.append(canonical_state((b1, b2)))
    # 去重（因为canonical_state可能使不同组合变同一tuple）
    states = list(dict.fromkeys(states))
    return states


# -----------------------------
# 轮内结束后的“选择带什么进入下一轮”：从候选球中选<=2个
# -----------------------------
def possible_next_carries(
    carry0: State,
    seq: Tuple[str, ...],
    carry_limit: int = 2,
) -> List[State]:
    """
    carry0：开局持有（未丢弃后的）
    seq：本轮抽到的颜色序列（长度N）
    新抽到的球 idx=抽到的位置(1..N)
    轮末可从(carry0 + 新球)中选<=2个作为next_carry
    """
    new_balls: List[Ball] = [(c, i + 1) for i, c in enumerate(seq)]
    candidates: List[Ball] = list(carry0) + new_balls

    # 选0/1/2个
    res: List[State] = []
    for k in range(0, carry_limit + 1):
        for comb in itertools.combinations(candidates, k):
            res.append(canonical_state(comb))

    # 去重（可能不同来源生成同一state）
    return list(dict.fromkeys(res))


# -----------------------------
# 轮开始前“可丢弃哪些”：drop子集
# -----------------------------
def possible_drops(carry: State) -> List[State]:
    """
    返回丢弃后的 carry0（即保留下来的那部分），等价于选择一个子集保留。
    """
    balls = list(carry)
    res: List[State] = []
    for k in range(0, len(balls) + 1):
        for keep in itertools.combinations(balls, k):
            res.append(canonical_state(keep))
    return list(dict.fromkeys(res))


# -----------------------------
# 核心：Bellman更新
# -----------------------------
def expected_cost_one_step(
    cfg: GameConfig,
    V: Dict[State, float],
    state: State,
) -> Tuple[float, State]:
    """
    返回 (该state的最优V值, 最优的carry0(=丢弃后保留的开局持有))。
    轮开始前可丢弃 -> 付费 -> 判定N -> 抽球序列 -> 若成功则0，否则轮末选next_carry最小V
    """
    target = cfg.target_colors
    base_w = cfg.weights
    colors_all = list(base_w.keys())

    best_val = math.inf
    best_carry0: State = state

    for carry0 in possible_drops(state):
        # 开局付费：1 + 当前持有数量
        cost = 1.0 + len(carry0)

        # 本轮开始时，袋子要剔除持有颜色（颜色层面剔除）
        removed_colors = {c for (c, _) in carry0}
        w2 = {c: w for c, w in base_w.items() if c not in removed_colors}
        if not w2:
            # 袋子空（理论上不会出现，因为你颜色>=7，carry<=2）
            continue
        w2 = normalize_weights(w2)

        # 删判定序号
        deleted = {idx for (_, idx) in carry0}
        distN = n_distribution_after_deletions(cfg.q, deleted)  # n=0..len(rem)

        # 开局已经持有的颜色集合
        start_colors = set(removed_colors)

        exp_future = 0.0
        for n, pn in enumerate(distN):
            if pn == 0.0:
                continue
            if n == 0:
                # 本轮不抽球
                # 只看是否开局就已达成目标（通常不会，因为目标<=3，carry<=2）
                if target.issubset(start_colors):
                    # 成功，未来成本0
                    continue
                # 失败：轮末可选择next_carry（此处seq为空）
                # 注意：你允许“继续持有”或“丢弃”，但丢弃发生在下一轮开始前，所以这里轮末可保持carry0不变
                candidates_next = possible_next_carries(carry0, (), cfg.carry_limit)
                min_next = min(V[s2] for s2 in candidates_next)
                exp_future += pn * min_next
                continue

            # n>0：枚举所有序列及概率
            seq_probs = enumerate_weighted_sequences(w2, n)
            for seq, ps in seq_probs.items():
                p_out = pn * ps

                # 过程中是否达成目标（任意时刻）
                if hits_target_during_round(start_colors, seq, target):
                    # 本轮成功，未来成本0
                    continue

                # 本轮未成功：轮末选择next_carry使未来期望最小
                candidates_next = possible_next_carries(carry0, seq, cfg.carry_limit)
                min_next = min(V[s2] for s2 in candidates_next)
                exp_future += p_out * min_next

        val = cost + exp_future
        if val < best_val:
            best_val = val
            best_carry0 = carry0

    return best_val, best_carry0


def solve_optimal_expected_tokens(cfg: GameConfig) -> Tuple[float, Dict[State, float], Dict[State, State]]:
    """
    价值迭代求解：
    - 返回：起始空状态的最小期望总代币、各状态V、以及“最优丢弃后开局持有carry0”策略
    """
    cfg = validate_cfg(cfg)
    colors = list(cfg.weights.keys())
    states = all_states(colors)

    # 初始化V：全设为0会偏乐观；全设为一个较大值也可。
    # 这里用0起步也能收敛（正成本、存在成功路径），但可能迭代次数略多。
    V = {s: 0.0 for s in states}
    policy_start: Dict[State, State] = {s: s for s in states}

    for it in range(cfg.vi_max_iter):
        max_delta = 0.0
        # Gauss-Seidel：按某顺序原地更新
        for s in states:
            new_v, best_carry0 = expected_cost_one_step(cfg, V, s)
            max_delta = max(max_delta, abs(new_v - V[s]))
            V[s] = new_v
            policy_start[s] = best_carry0

        if max_delta < cfg.vi_tol:
            break
    else:
        raise RuntimeError("价值迭代未在最大迭代次数内收敛；可调大vi_max_iter或放宽vi_tol")

    return V[()], V, policy_start


# -----------------------------
# 示例：你之前的9色 + 目标(红白蓝) + q=[1,0.5,0.3]
# -----------------------------
if __name__ == "__main__":
    weights_9 = {
        "red": 0.10,
        "white": 0.10,
        "blue": 0.12,
        "b1": 0.10,
        "b2": 0.10,
        "b3": 0.12,
        "b4": 0.12,
        "b5": 0.12,
        "b6": 0.12,
    }

    cfg = GameConfig(
        q=[1.0, 0.5, 0.3],
        weights=weights_9,
        target_colors={"red", "white", "blue"},  # Y=3
        carry_limit=2,
        vi_max_iter=20000,
        vi_tol=1e-12,
    )

    best, V, pol = solve_optimal_expected_tokens(cfg)
    print(f"最小期望总代币（从空手开始）= {best:.6f}")

    # 展示若干状态下“开局前最优保留(=丢弃后剩余)”策略
    samples = [
        (),
        (("red", 1),),
        (("b3", 2),),
        (("red", 1), ("white", 2)),
        (("b1", 1), ("b6", 3)),
    ]
    print("\n部分状态的最优开局保留策略（丢弃后剩余carry0）:")
    for s in samples:
        if s in pol:
            print(f"state={s}  ->  carry0={pol[s]}  | V={V[s]:.6f}")
