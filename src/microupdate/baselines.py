"""Baselines for the micro-update comparison — all on the same surrogate, budget and
candidate set as the RL agent (执行计划 §2.1 fair-evaluation-budget). A plan is a list
of (loc_idx, type_idx) placements with summed cost <= budget.

Implemented here: reference points (no_action, random, tree100), a domain rule
(heat_priority), the strongest comparator (greedy), and two metaheuristics (GA, SA) run
under a fixed surrogate-evaluation budget N_eval. PSO/ACO/BO extend the same interface
for the full 15-baseline E1 at scale.
"""
from __future__ import annotations

import numpy as np
from .action_space import ACTIONS, N_PLACEMENT_ACTIONS

TREE_LARGE = 2  # action id with strongest shade


# ---- plantability mask helpers (微更新现实约束) ------------------------------
def _ok(problem, li, ti):
    """Is action type ti allowed at candidate li? (crown-clearance / water mask)"""
    ok = getattr(problem, "action_ok", None)
    return True if ok is None else bool(ok[li, ti])


def _allowed_types(problem, li):
    return [ti for ti in range(N_PLACEMENT_ACTIONS) if _ok(problem, li, ti)]


def _largest_tree(problem, li):
    """Largest tree type allowed at li, or None if no tree fits (materials-only site)."""
    for ti in (TREE_LARGE, 1, 0):
        if _ok(problem, li, ti):
            return ti
    return None


def layout_from_plan(problem, placements):
    L = problem.empty_layout()
    for li, ti in placements:
        problem.place(L, li, ti)
    return L


def eval_plan(problem, placements):
    return problem.evaluate(layout_from_plan(problem, placements))["objective"]


def plan_metrics(problem, placements):
    m = problem.evaluate(layout_from_plan(problem, placements))
    m = {k: v for k, v in m.items() if k != "dutci_field"}
    return {**m, "cost": problem.plan_cost(placements), "n_actions": len(placements)}


# ---- reference points ------------------------------------------------------
def no_action(problem):
    return []


def tree100(problem):
    """Largest ALLOWED tree at every plantable candidate (ignore budget): shade-saturation ref."""
    plan = []
    for li in range(problem.K):
        ti = _largest_tree(problem, li)
        if ti is not None:
            plan.append((li, ti))
    return plan


def random_search(problem, n_eval=3000, rng=None):
    """Best random feasible plan over N_eval surrogate evaluations."""
    rng = rng or np.random.default_rng(0)
    best, best_obj = [], -1e9
    for _ in range(n_eval):
        plan = _random_feasible(problem, rng)
        obj = eval_plan(problem, plan)
        if obj > best_obj:
            best, best_obj = plan, obj
    return best


# ---- domain rule -----------------------------------------------------------
def heat_priority(problem, type_id=TREE_LARGE):
    """Fill hottest candidates first with the largest ALLOWED shade until budget
    (candidates are already sorted hottest-first; materials-only sites are skipped)."""
    plan, cost = [], 0.0
    for li in range(problem.K):
        ti = _largest_tree(problem, li)
        if ti is None:
            continue
        c = ACTIONS[ti].cost
        if cost + c > problem.budget:
            continue
        plan.append((li, ti)); cost += c
    return plan


def _seed_plans(problem):
    """Cheap domain-heuristic plans (each ~free to construct) used to WARM-START the
    metaheuristics — they then refine from a strong start instead of pure random, making
    the comparison genuinely competitive (no longer below-random). ~4 evals to score."""
    return [heat_priority(problem), equity_priority(problem),
            plantable_priority(problem), canopy_clustering(problem)]


def _floor(problem, plan):
    """Safety net: a search method never returns worse than the best cheap domain heuristic."""
    best = max(_seed_plans(problem), key=lambda p: eval_plan(problem, p))
    return plan if eval_plan(problem, plan) >= eval_plan(problem, best) else best


# ---- greedy (strongest comparator) -----------------------------------------
import heapq


def greedy(problem, verbose=False):
    """Lazy (accelerated) greedy: each step commit the (loc,type) with the best marginal
    objective gain per cost. Cooling has diminishing returns (shade saturation), so the
    lazy-greedy upper-bound trick gives the same plan as full greedy at ~30× fewer
    surrogate evaluations. This is the strongest, fairest comparator for RL."""
    used = np.zeros(problem.K, bool)
    plan, cost = [], 0.0
    cur_obj = problem.evaluate(problem.empty_layout())["objective"]
    # initial marginal gain/cost for every ALLOWED (loc,type), one batched pass
    cands = [(li, ti) for li in range(problem.K) for ti in range(N_PLACEMENT_ACTIONS)
             if _ok(problem, li, ti)]
    res = problem.evaluate_batch([layout_from_plan(problem, [c]) for c in cands])
    heap = [(-(res[k]["objective"] - cur_obj) / ACTIONS[cands[k][1]].cost,
             cands[k][0], cands[k][1], 0) for k in range(len(cands))]
    heapq.heapify(heap)
    step = 0
    while heap:
        neg_g, li, ti, last = heapq.heappop(heap)
        if used[li] or cost + ACTIONS[ti].cost > problem.budget:
            continue
        if last == step:                      # gain is current -> commit
            if -neg_g <= 0:
                break
            plan.append((li, ti)); used[li] = True; cost += ACTIONS[ti].cost
            cur_obj = problem.evaluate(layout_from_plan(problem, plan))["objective"]
            step += 1
        else:                                 # stale -> recompute this candidate's gain
            g = (problem.evaluate(layout_from_plan(problem, plan + [(li, ti)]))["objective"]
                 - cur_obj) / ACTIONS[ti].cost
            if g > 0:
                heapq.heappush(heap, (-g, li, ti, step))
    return plan


def budget_greedy(problem, seed=None, n_eval=256, rng=None):
    """Lazy-greedy bounded to a HARD surrogate-eval budget n_eval — the FAIR equal-budget protocol
    every search method is supposed to share (plain greedy() does an unbounded all-candidate scan =
    thousands of evals, an unfair advantage). Optionally SEEDED from an initial plan (e.g. the RL
    policy rollout): RL supplies the global structure, this refines/fills the rest within the shared
    budget. The initial marginal-gain scan is restricted to the hottest candidate cells so it fits
    the budget. Used both as the fair `greedy` (seed=[]) and as RL's learned-init local refinement."""
    import heapq
    rng = rng or np.random.default_rng(0)
    plan = list(seed) if seed else []
    used = np.zeros(problem.K, bool)
    for li, _ in plan:
        if 0 <= li < problem.K:
            used[li] = True
    cost = problem.plan_cost(plan)
    cheapest = min(a.cost for a in ACTIONS)
    cur = problem.evaluate(layout_from_plan(problem, plan))["objective"]; evals = 1
    if cost + cheapest > problem.budget:
        return plan
    bu = problem.baseline_utci[problem.cand[:problem.K, 0], problem.cand[:problem.K, 1]]
    order = np.argsort(-bu)                              # hottest cells first
    cands = [(int(li), ti) for li in order for ti in range(N_PLACEMENT_ACTIONS)
             if not used[int(li)] and _ok(problem, int(li), ti)]
    if not cands:
        return plan
    init_n = min(len(cands), max(8, n_eval // 2))        # ~half the budget seeds the heap
    cands = cands[:init_n]
    res = problem.evaluate_batch([layout_from_plan(problem, plan + [c]) for c in cands]); evals += init_n
    heap = [(-(res[k]["objective"] - cur) / ACTIONS[cands[k][1]].cost, cands[k][0], cands[k][1], 0)
            for k in range(len(cands))]
    heapq.heapify(heap); step = 0
    while heap and evals < n_eval and cost + cheapest <= problem.budget:
        neg_g, li, ti, last = heapq.heappop(heap)
        if used[li] or cost + ACTIONS[ti].cost > problem.budget:
            continue
        if last == step:
            if -neg_g <= 0:
                break
            plan.append((li, ti)); used[li] = True; cost += ACTIONS[ti].cost
            cur = problem.evaluate(layout_from_plan(problem, plan))["objective"]; evals += 1; step += 1
        else:
            g = (problem.evaluate(layout_from_plan(problem, plan + [(li, ti)]))["objective"] - cur) \
                / ACTIONS[ti].cost
            evals += 1
            if g > 0:
                heapq.heappush(heap, (-g, li, ti, step))
    return plan


# ---- metaheuristics --------------------------------------------------------
def _random_feasible(problem, rng):
    order = rng.permutation(problem.K)
    plan, cost = [], 0.0
    for li in order:
        types = _allowed_types(problem, int(li))
        if not types:
            continue
        ti = int(types[int(rng.integers(len(types)))])
        if rng.random() < 0.5:      # ~half the candidates used on average
            continue
        if cost + ACTIONS[ti].cost > problem.budget:
            continue
        plan.append((int(li), ti)); cost += ACTIONS[ti].cost
    return plan


def _repair(problem, genome):
    """genome: dict loc->type. Drop placements (cheapest-gain-last order = random) until
    within budget."""
    items = list(genome.items())
    cost = sum(ACTIONS[t].cost for _, t in items)
    rng = np.random.default_rng(len(items))
    while cost > problem.budget and items:
        i = int(rng.integers(len(items)))
        loc, t = items.pop(i); cost -= ACTIONS[t].cost
        genome.pop(loc, None)
    return genome


def ga(problem, n_eval=3000, pop=40, rng=None):
    rng = rng or np.random.default_rng(1)
    def to_genome(plan): return {li: ti for li, ti in plan}
    def to_plan(g): return list(g.items())
    population = [to_genome(_random_feasible(problem, rng)) for _ in range(pop)]
    for i, sp in enumerate(_seed_plans(problem)):          # warm-start with domain heuristics
        if i < pop:
            population[i] = to_genome(sp)
    fitness = [eval_plan(problem, to_plan(g)) for g in population]
    evals = pop
    while evals < n_eval:
        # tournament selection + uniform crossover + mutation
        order = np.argsort(fitness)
        elite = [population[i] for i in order[-max(2, pop // 5):]]
        newpop = list(elite)
        while len(newpop) < pop:
            pa, pb = rng.choice(len(population), 2, replace=False)
            keys = set(population[pa]) | set(population[pb])
            child = {}
            for k in keys:
                src = population[pa] if (k in population[pa] and (k not in population[pb] or rng.random() < 0.5)) else population[pb]
                if k in src and rng.random() < 0.9:
                    child[k] = src[k]
            # mutation (allowed types only)
            if rng.random() < 0.6:
                loc = int(rng.integers(problem.K)); types = _allowed_types(problem, loc)
                if types:
                    child[loc] = int(types[int(rng.integers(len(types)))])
            if child and rng.random() < 0.3:
                child.pop(int(rng.choice(list(child))), None)
            newpop.append(_repair(problem, child))
        population = newpop
        fitness = [eval_plan(problem, to_plan(g)) for g in population]
        evals += pop
    best = population[int(np.argmax(fitness))]
    return _floor(problem, to_plan(best))


def pso(problem, n_eval=3000, swarm=30, rng=None):
    """Particle-swarm over a continuous score matrix S[K,n_types]; each particle decodes
    to a plan by greedy budget-fill on its scores. (Strategic-tree-placement 2024 uses PSO.)"""
    rng = rng or np.random.default_rng(3)
    K, nt = problem.K, N_PLACEMENT_ACTIONS

    def decode(score):
        flat = score.ravel()
        order = np.argsort(-flat)
        used = np.zeros(K, bool); plan = []; cost = 0.0
        for idx in order:
            if flat[idx] <= 0:
                break
            li, ti = divmod(int(idx), nt)
            if used[li] or not _ok(problem, li, ti) or cost + ACTIONS[ti].cost > problem.budget:
                continue
            plan.append((li, ti)); used[li] = True; cost += ACTIONS[ti].cost
        return plan

    pos = rng.normal(0, 1, (swarm, K, nt)); vel = np.zeros_like(pos)
    pbest = pos.copy(); pbest_f = np.array([eval_plan(problem, decode(p)) for p in pos])
    evals = swarm
    g = int(pbest_f.argmax()); gbest = pbest[g].copy(); gbest_f = float(pbest_f[g])
    w, c1, c2 = 0.7, 1.5, 1.5
    while evals < n_eval:
        for i in range(swarm):
            vel[i] = (w * vel[i] + c1 * rng.random() * (pbest[i] - pos[i])
                      + c2 * rng.random() * (gbest - pos[i]))
            pos[i] += vel[i]
            f = eval_plan(problem, decode(pos[i])); evals += 1
            if f > pbest_f[i]:
                pbest[i] = pos[i].copy(); pbest_f[i] = f
            if f > gbest_f:
                gbest, gbest_f = pos[i].copy(), f
            if evals >= n_eval:
                break
    return _floor(problem, decode(gbest))


def aco(problem, n_eval=3000, ants=20, rng=None):
    """Ant-colony: pheromone τ[K,n_types] + heat heuristic η; ants build plans, best
    reinforce τ. (Strategic-tree-placement 2024 uses ACO.)"""
    rng = rng or np.random.default_rng(4)
    K, nt = problem.K, N_PLACEMENT_ACTIONS
    heat = problem.baseline_utci[problem.cand[:K, 0], problem.cand[:K, 1]]   # actual K (cand is padded to K_max)
    eta = np.tile((heat - heat.min() + 1e-3)[:, None], (1, nt))
    okmat = np.array([[_ok(problem, li, ti) for ti in range(nt)] for li in range(K)], dtype=float)
    eta = eta * okmat                                   # disallowed (loc,type) -> zero probability
    tau = np.ones((K, nt)); best, best_f = [], -1e9; evals = 0; rho = 0.1
    while evals < n_eval:
        sols = []
        for _ in range(ants):
            prob = tau * eta ** 2
            used = np.zeros(K, bool); plan = []; cost = 0.0
            for li in rng.permutation(K):
                p = prob[li]
                if p.sum() <= 0:
                    continue
                ti = int(rng.choice(nt, p=p / p.sum()))
                if cost + ACTIONS[ti].cost > problem.budget or rng.random() < 0.4:
                    continue
                plan.append((int(li), ti)); used[li] = True; cost += ACTIONS[ti].cost
            f = eval_plan(problem, plan); evals += 1; sols.append((f, plan))
            if f > best_f:
                best_f, best = f, plan
            if evals >= n_eval:
                break
        tau *= (1 - rho)
        for f, plan in sols:
            for li, ti in plan:
                tau[li, ti] += max(f, 0) / (abs(best_f) + 1e-6)
    return _floor(problem, best)


def sa(problem, n_eval=3000, rng=None):
    rng = rng or np.random.default_rng(2)
    seed0 = max(_seed_plans(problem), key=lambda p: eval_plan(problem, p))   # warm-start
    cur = {li: ti for li, ti in seed0}
    cur_obj = eval_plan(problem, list(cur.items()))
    best, best_obj = dict(cur), cur_obj
    T0, T1 = 0.5, 0.01
    for it in range(n_eval - 1):
        T = T0 * (T1 / T0) ** (it / max(1, n_eval - 1))
        cand = dict(cur)
        move = rng.random()
        if move < 0.4 and cand:                       # remove
            cand.pop(int(rng.choice(list(cand))), None)
        elif move < 0.7:                              # add (allowed types only)
            loc = int(rng.integers(problem.K)); types = _allowed_types(problem, loc)
            if types:
                cand[loc] = int(types[int(rng.integers(len(types)))])
        else:                                         # change type (allowed types only)
            loc = int(rng.integers(problem.K)); types = _allowed_types(problem, loc)
            if types:
                cand[loc] = int(types[int(rng.integers(len(types)))])
        _repair(problem, cand)
        obj = eval_plan(problem, list(cand.items()))
        if obj > cur_obj or rng.random() < np.exp((obj - cur_obj) / max(T, 1e-6)):
            cur, cur_obj = cand, obj
            if obj > best_obj:
                best, best_obj = dict(cand), obj
    return _floor(problem, list(best.items()))


# ===== full 15-baseline set: remaining domain rules, NSGA-II, BO, best-found =====

def plantable_priority(problem):
    """可种空间优先: fill the most plantable (open ground) cells first with medium trees."""
    K = problem.K; dens = np.empty(K)
    for i in range(K):
        r, c = int(problem.cand[i, 0]), int(problem.cand[i, 1])
        r0, r1 = max(0, r - 2), min(problem.H, r + 3)
        c0, c1 = max(0, c - 2), min(problem.W, c + 3)
        dens[i] = problem.ground[r0:r1, c0:c1].sum()
    plan, cost = [], 0.0
    for i in np.argsort(-dens):
        ti = 1 if _ok(problem, int(i), 1) else (0 if _ok(problem, int(i), 0) else None)
        if ti is None:
            continue
        c = ACTIONS[ti].cost
        if cost + c > problem.budget:
            continue
        plan.append((int(i), ti)); cost += c
    return plan


def equity_priority(problem):
    """公平/风险优先: reach as MANY high-risk (hot) cells as possible with cheap small trees."""
    K = problem.K
    heat = problem.baseline_utci[problem.cand[:K, 0], problem.cand[:K, 1]]
    plan, cost, c0 = [], 0.0, ACTIONS[0].cost
    for i in np.argsort(-heat):
        if not _ok(problem, int(i), 0):
            continue
        if cost + c0 > problem.budget:
            break
        plan.append((int(i), 0)); cost += c0
    return plan


def canopy_clustering(problem, radius=3):
    """树冠聚集: grow contiguous tree clusters around hottest seeds (additive-shade hypothesis)."""
    K = problem.K
    coords = problem.cand[:K]
    heat = problem.baseline_utci[coords[:, 0], coords[:, 1]]
    plan, cost, used = [], 0.0, set()
    for seed in np.argsort(-heat):
        if int(seed) in used:
            continue
        sr, sc = int(coords[seed, 0]), int(coords[seed, 1])
        near = [j for j in range(K) if j not in used
                and abs(int(coords[j, 0]) - sr) <= radius and abs(int(coords[j, 1]) - sc) <= radius]
        near.sort(key=lambda j: abs(int(coords[j, 0]) - sr) + abs(int(coords[j, 1]) - sc))
        for j in near:
            tj = 1 if _ok(problem, j, 1) else (0 if _ok(problem, j, 0) else None)
            if tj is None:
                used.add(j)
                continue
            ct = ACTIONS[tj].cost
            if cost + ct > problem.budget:
                return plan
            plan.append((int(j), tj)); used.add(j); cost += ct
    return plan


def nsga2(problem, n_eval=3000, pop=40, rng=None):
    """Compact NSGA-II on (maximise hotspot cooling, minimise cost); return the within-budget
    solution with the best cooling."""
    rng = rng or np.random.default_rng(5)

    def evalp(plan):
        m = problem.evaluate(layout_from_plan(problem, plan))
        return m["mean_cool_hot"], problem.plan_cost(plan)

    def dominates(a, b):
        return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])

    P = [_random_feasible(problem, rng) for _ in range(pop)]
    for i, sp in enumerate(_seed_plans(problem)):          # warm-start with domain heuristics
        if i < pop:
            P[i] = sp
    F = [evalp(p) for p in P]; evals = pop
    while evals < n_eval:
        Q = []
        for p in P:
            cand = {li: ti for li, ti in p}
            for _ in range(2):
                if rng.random() < 0.5 and cand:
                    cand.pop(int(rng.choice(list(cand))), None)
                else:
                    loc = int(rng.integers(problem.K)); types = _allowed_types(problem, loc)
                    if types:
                        cand[loc] = int(types[int(rng.integers(len(types)))])
            _repair(problem, cand); Q.append(list(cand.items()))
        R = P + Q; FR = F + [evalp(q) for q in Q]; evals += len(Q)
        idx = sorted(range(len(R)),
                     key=lambda i: (sum(dominates(FR[j], FR[i]) for j in range(len(R))), -FR[i][0]))
        P = [R[i] for i in idx[:pop]]; F = [FR[i] for i in idx[:pop]]
    feas = [(p, f) for p, f in zip(P, F) if f[1] <= problem.budget]
    return _floor(problem, max(feas, key=lambda pf: pf[1][0])[0] if feas else P[0])


def bo(problem, n_eval=400, rng=None):
    """Sample-efficient search over a compact 6-d parameterisation (budget share per action
    type, spent at hottest free candidates) — a lightweight Bayesian-style optimiser (no GP dep)."""
    rng = rng or np.random.default_rng(6)
    K = problem.K
    heat_order = np.argsort(-problem.baseline_utci[problem.cand[:K, 0], problem.cand[:K, 1]])

    def decode(shares):
        shares = np.clip(shares, 0, None); shares = shares / (shares.sum() + 1e-9)
        plan, used, cost = [], set(), 0.0
        for ti in np.argsort(-shares):
            cap = shares[ti] * problem.budget; spent = 0.0; ct = ACTIONS[ti].cost
            for li in heat_order:
                if int(li) in used or not _ok(problem, int(li), int(ti)):
                    continue
                if spent + ct > cap or cost + ct > problem.budget:
                    continue
                plan.append((int(li), int(ti))); used.add(int(li)); spent += ct; cost += ct
        return plan

    best, best_obj, best_x = [], -1e9, None
    for _ in range(n_eval):
        if best_x is None or rng.random() < 0.3:
            x = rng.random(N_PLACEMENT_ACTIONS)
        else:
            x = np.clip(best_x + rng.normal(0, 0.15, N_PLACEMENT_ACTIONS), 0, None)
        plan = decode(x); obj = eval_plan(problem, plan)
        if obj > best_obj:
            best, best_obj, best_x = plan, obj, x
    return best


def best_found(problem, n_eval=5000, rng=None):
    """经验最优参照 (执行计划 参照点): greedy + long GA + bounded local search; the best plan
    found. Estimates the optimality gap — NOT a fair method (uses far more search than N_eval)."""
    rng = rng or np.random.default_rng(7)
    seed = max([greedy(problem), ga(problem, n_eval=n_eval, rng=rng)], key=lambda p: eval_plan(problem, p))
    cur = {li: ti for li, ti in seed}; best_obj = eval_plan(problem, list(cur.items()))
    for _ in range(n_eval // 2):
        trial = dict(cur); m = rng.random()
        if m < 0.4 and trial:
            trial.pop(int(rng.choice(list(trial))), None)
        elif m < 0.7:
            loc = int(rng.integers(problem.K)); types = _allowed_types(problem, loc)
            if types:
                trial[loc] = int(types[int(rng.integers(len(types)))])
        elif trial:
            loc = int(rng.choice(list(trial))); types = _allowed_types(problem, loc)
            if types:
                trial[loc] = int(types[int(rng.integers(len(types)))])
        _repair(problem, trial)
        o = eval_plan(problem, list(trial.items()))
        if o > best_obj:
            cur, best_obj = trial, o
    return list(cur.items())
