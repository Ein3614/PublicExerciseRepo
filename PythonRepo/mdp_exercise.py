from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Set, Iterable
import itertools
import math

Color = str
State = Tuple[Optional[Color], Optional[Color], Optional[Color]]  # (slot1, slot2, slot3)


@dataclass(frozen=True)
class GameConfig:
    # q1,q2,q3：分别对应 slot1/2/3 为空时“能捞到球”的概率
    q: Tuple[float, float, float]
    # 初始颜色权重（不要求归一化）
    weights: Dict[Color, float]
    # 目标颜色集合（<=3）
    target: Set[Color]
    # 每轮开始前最多保留多少个球（你的规则=2）
    keep_limit: int = 2
    # 价值迭代
    vi_max_iter: int = 50000
    vi_tol: float = 1e-12


def normalize(w: Dict[Color, float]) -> Dict[Color, float]:
    for k, v in w.items():
        if v < 0:
            raise ValueError(f"negative weight: {k}={v}")
    s = sum(w.values())
    if s <= 0:
        raise ValueError("sum of weights must be > 0")
    return {k: v / s for k, v in w.items()}


def validate(cfg: GameConfig) -> GameConfig:
    if len(cfg.q) != 3:
        raise ValueError("q must have length 3")
    for i, p in enumerate(cfg.q, start=1):
        if not (0 <= p <= 1):
            raise ValueError(f"q{i} out of range: {p}")
    if not cfg.target or len(cfg.target) > 3:
        raise ValueError("target must be non-empty and size<=3")
    w = normalize(dict(cfg.weights))
    if any(t not in w for t in cfg.target):
        missing = cfg.target - set(w.keys())
        raise ValueError(f"target contains unknown colors: {missing}")
    if cfg.keep_limit != 2:
        raise ValueError("this template assumes keep_limit=2")
    return GameConfig(
        q=cfg.q,
        weights=w,
        target=set(cfg.target),
        keep_limit=cfg.keep_limit,
        vi_max_iter=cfg.vi_max_iter,
        vi_tol=cfg.vi_tol,
    )


# ---------- 状态空间：slot模型 ----------
def all_states(colors: List[Color]) -> List[State]:
    """
    枚举 (s1,s2,s3) 其中每个slot为空或装某颜色，且不允许重复颜色。
    """
    states: List[State] = []
    # k=0..3 选择k种颜色放入k个slot
    for k in range(0, 4):
        for chosen_colors in itertools.permutations(colors, k):
            # 选k个slot位置
            for slots in itertools.combinations([0, 1, 2], k):
                s: List[Optional[Color]] = [None, None, None]
                ok = True
                used = set()
                for pos, c in zip(slots, chosen_colors):
                    if c in used:
                        ok = False
                        break
                    used.add(c)
                    s[pos] = c
                if ok:
                    states.append((s[0], s[1], s[2]))
    # 去重（上面生成方式已基本不重复，这里保险）
    return list(dict.fromkeys(states))


def held_colors(state: State) -> Set[Color]:
    return {c for c in state if c is not None}


def is_success(state: State, target: Set[Color]) -> bool:
    return target.issubset(held_colors(state))


# ---------- 轮开始前动作：选择保留哪些slot（最多2个） ----------
def possible_keeps(state: State, keep_limit: int) -> List[Tuple[State, float]]:
    """
    返回所有可能的“丢弃后状态 keep_state”，以及本轮固定成本 cost = 1 + kept_count
    keep_state：只保留选中的slot，其他slot清空
    """
    filled_slots = [i for i, c in enumerate(state) if c is not None]
    res: List[Tuple[State, float]] = []
    for k in range(0, min(keep_limit, len(filled_slots)) + 1):
        for keep_slots in itertools.combinations(filled_slots, k):
            s = [None, None, None]
            for i in keep_slots:
                s[i] = state[i]
            cost = 1.0 + k
            res.append(((s[0], s[1], s[2]), cost))
    return list(dict.fromkeys(res))


# ---------- 空slot判定：哪些slot成功得到球 ----------
def success_slot_subsets(empty_slots: List[int], q: Tuple[float, float, float]) -> Dict[Tuple[int, ...], float]:
    """
    返回：成功slot集合(升序tuple) -> 概率
    """
    out: Dict[Tuple[int, ...], float] = {}
    for r in range(0, len(empty_slots) + 1):
        for succ in itertools.combinations(empty_slots, r):
            p = 1.0
            succ_set = set(succ)
            for i in empty_slots:
                pi = q[i]
                p *= pi if i in succ_set else (1 - pi)
            out[tuple(sorted(succ))] = out.get(tuple(sorted(succ)), 0.0) + p
    return out


# ---------- 从袋子按“无放回”填充若干slot：枚举所有结果及概率 ----------
def enumerate_fill_sequences(
    base_weights: Dict[Color, float],
    start_state: State,
    slots_to_fill: Tuple[int, ...],
) -> Dict[Tuple[State, bool], float]:
    """
    对指定的 slots_to_fill（按slot从小到大顺序），
    枚举所有可能填充结果 state_end 及其概率；
    同时返回是否在填充过程中达成目标的标志（这里只返回False，成功判断在外层做更灵活）
    """
    # 这里不直接判断成功，让外层按“任意时刻达成”来判
    out: Dict[Tuple[State, bool], float] = {}

    def rec(curr_state: List[Optional[Color]], idx: int, prob: float):
        if idx == len(slots_to_fill):
            out[(tuple(curr_state), False)] = out.get((tuple(curr_state), False), 0.0) + prob
            return

        # 当前袋子可用颜色 = 初始权重去掉“当前持有颜色”
        removed = {c for c in curr_state if c is not None}
        w = {c: w for c, w in base_weights.items() if c not in removed and w > 0}
        if not w:
            # 没得抽了（一般不会发生）
            out[(tuple(curr_state), False)] = out.get((tuple(curr_state), False), 0.0) + prob
            return
        w = normalize(w)
        total = 1.0  # 已归一化

        slot = slots_to_fill[idx]
        for c, pc in w.items():
            next_state = curr_state[:]
            next_state[slot] = c
            rec(next_state, idx + 1, prob * pc)

    rec(list(start_state), 0, 1.0)
    return out


# ---------- 一步期望：从state出发，本轮最优保留策略下的最小期望 ----------
def bellman_one(cfg: GameConfig, V: Dict[State, float], state: State) -> Tuple[float, State]:
    target = cfg.target
    if is_success(state, target):
        return 0.0, state  # 已成功：未来成本0（不再开局）

    best_val = math.inf
    best_keep_state = state

    # 轮开始前选择保留（丢弃其余）
    for keep_state, fixed_cost in possible_keeps(state, cfg.keep_limit):
        # 若保留后已经集齐目标，直接成功（不用开局？按你的流程：开局前就满足了就不需要开局）
        if is_success(keep_state, target):
            val = 0.0
            if val < best_val:
                best_val = val
                best_keep_state = keep_state
            continue

        empty = [i for i, c in enumerate(keep_state) if c is None]
        # 对空slot做判定：哪些slot会捞到球
        succ_sets = success_slot_subsets(empty, cfg.q)

        exp_future = 0.0
        for slots_filled, p_slots in succ_sets.items():
            if p_slots == 0:
                continue

            # 根据 slots_filled 从袋子抽球填入（无放回），枚举所有填充结果
            fills = enumerate_fill_sequences(cfg.weights, keep_state, slots_filled)
            for (end_state, _), p_fill in fills.items():
                p_out = p_slots * p_fill

                # 在“填充过程中任意时刻”达成目标：需要按slot顺序逐步检查
                # 我们重放一次“按slot顺序填入”的过程来检查是否中途成功
                # （n<=3，开销很小）
                curr = list(keep_state)
                success = False
                if target.issubset({c for c in curr if c is not None}):
                    success = True
                for sidx in slots_filled:
                    curr[sidx] = end_state[sidx]
                    if target.issubset({c for c in curr if c is not None}):
                        success = True
                        break

                if success:
                    # 成功吸收：未来成本0
                    continue
                exp_future += p_out * V[end_state]

        val = fixed_cost + exp_future
        if val < best_val:
            best_val = val
            best_keep_state = keep_state

    return best_val, best_keep_state


def solve(cfg: GameConfig) -> Tuple[float, Dict[State, float], Dict[State, State]]:
    cfg = validate(cfg)
    colors = list(cfg.weights.keys())
    states = all_states(colors)

    V: Dict[State, float] = {s: 0.0 for s in states}
    policy: Dict[State, State] = {s: s for s in states}

    for _ in range(cfg.vi_max_iter):
        max_delta = 0.0
        for s in states:
            new_v, best_keep_state = bellman_one(cfg, V, s)
            max_delta = max(max_delta, abs(new_v - V[s]))
            V[s] = new_v
            policy[s] = best_keep_state
        if max_delta < cfg.vi_tol:
            break
    else:
        raise RuntimeError("value iteration did not converge; increase vi_max_iter or relax vi_tol")

    start: State = (None, None, None)
    return V[start], V, policy


# ---------- 可读性增强：state/action 表示 + 表格输出 ----------

def fmt_slot(c: Optional[str]) -> str:
    return "-" if c is None else str(c)

def fmt_state(state: State) -> str:
    # (s1,s2,s3)
    return f"[1:{fmt_slot(state[0])} | 2:{fmt_slot(state[1])} | 3:{fmt_slot(state[2])}]"

def target_progress(state: State, target: Set[str]) -> str:
    held = {c for c in state if c is not None}
    got = held & set(target)
    return f"{len(got)}/{len(target)}"

def keep_slots(from_state: State, keep_state: State) -> Tuple[int, ...]:
    """
    policy 里存的是 keep_state（丢弃后保留的状态）
    我们把它转成“保留了哪些slot编号”
    """
    slots = []
    for i in range(3):
        if keep_state[i] is not None:
            # keep_state 的slot必然来自 from_state 的同slot（因为保留=不清空该slot）
            slots.append(i + 1)
    return tuple(slots)

def fixed_cost_from_keep(keep_state: State) -> int:
    return 1 + sum(1 for c in keep_state if c is not None)

def print_policy_table(
    cfg: GameConfig,
    V: Dict[State, float],
    policy: Dict[State, State],
    *,
    sort_by: str = "V",          # "V" 或 "progress" 或 "filled"
    limit: Optional[int] = 60,   # None 表示全量输出
    only_reachable_style: bool = False,  # 预留参数：若未来做可达性筛选
) -> None:
    """
    以markdown表格形式打印策略：
    state | progress | best_keep_slots | fixed_cost | V(state)
    """
    target = cfg.target
    rows = []
    for s, ks in policy.items():
        prog = target_progress(s, target)
        keep = keep_slots(s, ks)
        cost = fixed_cost_from_keep(ks)
        filled = sum(1 for x in s if x is not None)
        rows.append((s, prog, filled, keep, cost, V[s]))

    if sort_by == "V":
        rows.sort(key=lambda x: x[5])
    elif sort_by == "progress":
        # 先按已拥有目标色数降序，再按V升序
        rows.sort(key=lambda x: (-int(x[1].split("/")[0]), x[5]))
    elif sort_by == "filled":
        rows.sort(key=lambda x: (x[2], x[5]))
    else:
        raise ValueError("sort_by must be one of: V, progress, filled")

    if limit is not None:
        rows = rows[:limit]

    # 打印markdown表格
    print("| state (slots) | target progress | filled slots | best keep slots | fixed cost | V(state) |")
    print("|---|---:|---:|---:|---:|---:|")
    for s, prog, filled, keep, cost, val in rows:
        print(f"| {fmt_state(s)} | {prog} | {filled} | {keep} | {cost} | {val:.6f} |")



# best, V, pol = solve(cfg)
# print(f"从空盒开始的最小期望总代币 = {best:.6f}\n")

# print("=== 按 V(state) 从小到大展示（前60行）===")
# print_policy_table(cfg, V, pol, sort_by="V", limit=60)

# print("\n=== 优先看接近完成的状态（progress降序）===")
# print_policy_table(cfg, V, pol, sort_by="progress", limit=60)


# -----------------------------
# 示例：9色 + 目标红白蓝 + q=[1,0.5,0.3]
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
        q=(1.0, 0.5, 0.3),
        weights=weights_9,
        target={"red", "white", "blue"},
        keep_limit=2,
        vi_max_iter=30000,
        vi_tol=1e-12,
    )

    best, V, pol = solve(cfg)
    print(f"从空盒开始的最小期望总代币 = {best:.6f}")

    # 打印一些状态的“开局前保留哪些slot”策略
    samples: List[State] = [
        (None, None, None),
        ("red", None, None),
        ("red", "white", None),
        ("b3", "b6", "b1"),   # 三盒全满（轮与轮之间允许存在）
        ("red", "b6", "white")
    ]
    print("\n部分状态的最优“丢弃后保留状态”（开局前动作）:")
    for s in samples:
        if s in pol:
            print(f"{s} -> keep {pol[s]} | V={V[s]:.6f}")
