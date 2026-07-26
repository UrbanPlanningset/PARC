"""Tile-agnostic (generalist) Dueling Double DQN — ONE policy per CITY.

Per 执行计划 (每城独立): train a single policy on a city's tiles, then apply it to ALL of
that city's tiles by inference (cheap). The Q-net takes candidate coords PER FORWARD (not
fixed at construction), and TileProblem pads candidates to a fixed K_max, so any tile's
state is a valid input to the same network. Memory-slim replay (STATE_HW=64, cap 3000).
"""
from __future__ import annotations

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_space import N_PLACEMENT_ACTIONS, ACTIONS
from .env import MicroUpdateEnv


class GeneralistQNet(nn.Module):
    """Fully-conv dueling Q-net. forward(raster, scalar, cand) — cand:(B,K_max,2) in the
    state grid — gathers a per-(type,cell) advantage map at the tile's candidates, plus a
    global V(s) and STOP head. Action index = loc*n_types + type (loc-major), STOP last."""

    def __init__(self, in_ch=7, n_scalar=2 + N_PLACEMENT_ACTIONS + 4, n_types=N_PLACEMENT_ACTIONS,
                 hidden=64, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.n_types = n_types
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.scalar = nn.Linear(n_scalar, hidden)
        self.adv = nn.Conv2d(hidden, n_types, 1)
        self.val = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.stop = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, raster, scalar, cand):
        B = raster.shape[0]
        f = self.enc(raster)
        f = f + self.scalar(scalar).unsqueeze(-1).unsqueeze(-1)
        adv_map = self.adv(f)                                  # (B, n_types, hs, ws)
        hs, ws = adv_map.shape[2], adv_map.shape[3]
        flat = adv_map.reshape(B, self.n_types, hs * ws)
        idx = (cand[:, :, 0] * ws + cand[:, :, 1]).long()      # (B, K_max)
        Kx = idx.shape[1]
        idx_exp = idx.unsqueeze(1).expand(B, self.n_types, Kx)
        adv_cand = torch.gather(flat, 2, idx_exp)              # (B, n_types, K_max)
        adv_cand = adv_cand.permute(0, 2, 1).reshape(B, Kx * self.n_types)   # loc-major
        pooled = f.mean(dim=(2, 3))
        adv = torch.cat([adv_cand, self.stop(pooled)], dim=1)  # (B, K_max*n_types + 1)
        if not self.dueling:
            return adv
        return self.val(pooled) + (adv - adv.mean(dim=1, keepdim=True))


class FlatQNet(nn.Module):
    """ABLATION (no spatial structure): per-candidate MLP over [the candidate cell's own input
    features + a global-pooled context + scalars]. NO convolution / neighbourhood — so it cannot
    learn spatial allocation or coherent layouts. Same forward signature & output shape as
    GeneralistQNet (B, K_max*n_types + 1), so it is a drop-in replacement to isolate the value of
    the spatial Q-net."""

    def __init__(self, in_ch=7, n_scalar=2 + N_PLACEMENT_ACTIONS + 4, n_types=N_PLACEMENT_ACTIONS,
                 hidden=64, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.n_types = n_types
        self.in_ch = in_ch
        self.ctx = nn.Sequential(nn.Linear(in_ch + n_scalar, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.adv = nn.Sequential(nn.Linear(in_ch + hidden, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, n_types))
        self.val = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.stop = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, raster, scalar, cand):
        B, C, H, W = raster.shape
        g = raster.mean(dim=(2, 3))                              # (B, in_ch) global pool (no spatial)
        ctx = self.ctx(torch.cat([g, scalar], dim=1))           # (B, hidden)
        idx = (cand[:, :, 0] * W + cand[:, :, 1]).long()        # (B, K)
        K = idx.shape[1]
        flat = raster.reshape(B, C, H * W)
        cell = torch.gather(flat, 2, idx.unsqueeze(1).expand(B, C, K)).permute(0, 2, 1)  # (B, K, C)
        ctx_k = ctx.unsqueeze(1).expand(B, K, ctx.shape[1])
        adv_cand = self.adv(torch.cat([cell, ctx_k], dim=2))    # (B, K, n_types)
        adv_cand = adv_cand.reshape(B, K * self.n_types)        # loc-major
        adv = torch.cat([adv_cand, self.stop(ctx)], dim=1)      # (B, K*n_types + 1)
        if not self.dueling:
            return adv
        return self.val(ctx) + (adv - adv.mean(dim=1, keepdim=True))


class Replay:
    def __init__(self, cap=3000):
        self.buf = deque(maxlen=cap)

    def push(self, *t):
        self.buf.append(t)

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


class GeneralistAgent:
    def __init__(self, device="cpu", gamma=0.97, lr=5e-4, dueling=True, double=True, flat=False):
        self.device = device
        self.double = double
        Net = FlatQNet if flat else GeneralistQNet     # flat=True -> non-spatial ablation net
        self.online = Net(dueling=dueling).to(device)
        self.target = Net(dueling=dueling).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = Replay()
        self.cost_bias = False           # cost-biased exploration -> learn 'spread many cheap'
        self._ew = None; self._ew_n = None

    def _tensify(self, states, cands):
        r = torch.from_numpy(np.stack([s["raster"] for s in states]).astype(np.float32)).to(self.device)
        s = torch.from_numpy(np.stack([s["scalar"] for s in states]).astype(np.float32)).to(self.device)
        c = torch.from_numpy(np.stack(cands)).to(self.device)
        return r, s, c

    def _explore_weights(self, n):
        """Cost-biased exploration weights over the full action vector: placement actions get
        weight 1/cost (cheap actions explored more, so coherent 'spread many cheap' rollouts
        get sampled into replay), STOP gets the smallest single weight (discourage premature
        stop -> longer rollouts that fill the budget). Cached by action-vector length."""
        if self._ew_n == n:
            return self._ew
        costs = np.array([a.cost for a in ACTIONS], dtype=np.float64)
        w = np.empty(n, dtype=np.float64)
        for i in range(n - 1):
            w[i] = 1.0 / costs[i % N_PLACEMENT_ACTIONS]
        w[n - 1] = 1.0 / costs.max()                       # STOP last, least likely
        self._ew_n, self._ew = n, w
        return w

    @torch.no_grad()
    def act(self, state, cand, mask, eps):
        if random.random() < eps:
            legal = np.flatnonzero(mask)
            if self.cost_bias and len(legal) > 1:
                wl = self._explore_weights(len(mask))[legal]
                return int(random.choices(legal.tolist(), weights=wl.tolist(), k=1)[0])
            return int(random.choice(legal))
        r, s, c = self._tensify([state], [cand])
        q = self.online(r, s, c).squeeze(0).cpu().numpy()
        q[~mask] = -1e9
        return int(q.argmax())

    @torch.no_grad()
    def act_boltzmann(self, state, cand, mask, tau, rng):
        """Boltzmann (temperature) action sampling over the LEGAL set: a ~ softmax(Q[legal]/tau).
        Concentrates exploration on high-Q actions (unlike cost-blind eps-greedy), so best-of-N
        samples genuinely different high-value plans instead of near-duplicate greedy ones."""
        r, s, c = self._tensify([state], [cand])
        q = self.online(r, s, c).squeeze(0).cpu().numpy()
        legal = np.flatnonzero(mask)
        if len(legal) == 1 or tau <= 1e-6:
            return int(legal[int(np.argmax(q[legal]))])
        z = q[legal] / tau
        z -= z.max()
        p = np.exp(z); tot = p.sum()
        if not np.isfinite(tot) or tot <= 0:
            return int(legal[int(np.argmax(q[legal]))])
        return int(rng.choice(legal, p=p / tot))

    def learn(self, batch_size=64):
        if len(self.replay) < batch_size * 4:
            return None
        batch = self.replay.sample(batch_size)
        states, cands, actions, rewards, next_states, dones, next_masks = zip(*batch)
        r, s, c = self._tensify(states, cands)
        nr, ns, nc = self._tensify(next_states, cands)          # same tile -> same cand
        a = torch.tensor(actions, device=self.device).long()
        rew = torch.tensor(rewards, device=self.device).float()
        done = torch.tensor(dones, device=self.device).float()
        nmask = torch.from_numpy(np.stack(next_masks)).to(self.device)
        q = self.online(r, s, c).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                nq_online = self.online(nr, ns, nc); nq_online[~nmask] = -1e9
                best = nq_online.argmax(dim=1, keepdim=True)
                nq_target = self.target(nr, ns, nc).gather(1, best).squeeze(1)
            else:
                nq = self.target(nr, ns, nc); nq[~nmask] = -1e9
                nq_target = nq.max(dim=1).values
            y = rew + self.gamma * nq_target * (1 - done)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync(self):
        self.target.load_state_dict(self.online.state_dict())


def train_city(envs, agent, episodes=400, eps_start=1.0, eps_end=0.05, eps_decay=0.99,
               sync_every=10, learn_every=2, rng=None, log=None):
    """Train ONE policy across a city's tiles: each episode samples a random tile."""
    rng = rng or np.random.default_rng(0)
    cand_grids = [e.cand_grid() for e in envs]
    eps = eps_start
    tick = 0
    for ep in range(episodes):
        ei = int(rng.integers(len(envs)))
        env, cand = envs[ei], cand_grids[ei]
        state = env.reset(); mask = env.legal_mask(); done = False; epR = 0.0
        while not done:
            action = agent.act(state, cand, mask, eps)
            nstate, reward, done, _ = env.step(action); nmask = env.legal_mask()
            agent.replay.push(state, cand, action, reward, nstate, done, nmask)
            tick += 1
            if tick % learn_every == 0:
                agent.learn()
            state, mask = nstate, nmask; epR += reward
        eps = max(eps_end, eps * eps_decay)
        if ep % sync_every == 0:
            agent.sync()
        if log and ep % 50 == 0:
            log(f"    ep {ep:4d}/{episodes} eps={eps:.2f} tile={ei} ep_R={epR:+.2f}")
    return agent


def optimize_tile(env, device="cpu", episodes=80, warm_state=None, learn_every=2,
                  eps_start=1.0, eps_end=0.05, eps_decay=0.96, sync_every=10, rng=None):
    """PER-TILE RL optimisation (执行计划: RL 对每个地块一轮一轮迭代,代理做温度反馈).

    Run DQN on ONE tile: each episode places interventions step by step, the surrogate
    ('fast SOLWEIG') scores the result as the reward, the agent improves. Track the BEST plan
    found across all episodes (+ a final greedy rollout). Optionally warm-start from a policy
    pretrained on representative tiles (much faster convergence). Returns (best_plan, metrics)."""
    from .baselines import layout_from_plan, plan_metrics
    rng = rng or np.random.default_rng(0)
    cand = env.cand_grid()
    agent = GeneralistAgent(device=device)
    if warm_state is not None:
        agent.online.load_state_dict(warm_state); agent.target.load_state_dict(warm_state)
    eps = eps_start; tick = 0
    best_plan, best_obj = [], -1e9
    for ep in range(episodes):
        state = env.reset(); mask = env.legal_mask(); done = False; plan = []
        while not done:
            a = agent.act(state, cand, mask, eps)
            ns, r, done, _ = env.step(a); nmask = env.legal_mask()
            if a != env.STOP:
                li, ti = divmod(a, N_PLACEMENT_ACTIONS); plan.append((li, ti))
            agent.replay.push(state, cand, a, r, ns, done, nmask)
            tick += 1
            if tick % learn_every == 0:
                agent.learn()
            state, mask = ns, nmask
        obj = env.prev_obj                        # objective of this episode's final layout
        if obj > best_obj:
            best_obj, best_plan = obj, list(plan)
        eps = max(eps_end, eps * eps_decay)
        if ep % sync_every == 0:
            agent.sync()
    # final exploitation rollout of the converged policy — keep whichever is better
    gplan, gm = apply_policy(env, agent)
    if gm["objective"] > best_obj:
        best_plan = gplan
    return best_plan, plan_metrics(env.p, best_plan)


@torch.no_grad()
def apply_policy(env, agent):
    """Greedy rollout of the trained policy on one tile -> (placements, metrics)."""
    cand = env.cand_grid()
    state = env.reset(); mask = env.legal_mask(); done = False
    placements = []
    while not done:
        r, s, c = agent._tensify([state], [cand])
        q = agent.online(r, s, c).squeeze(0).cpu().numpy(); q[~mask] = -1e9
        action = int(q.argmax())
        if action == env.STOP:
            break
        li, ti = divmod(action, N_PLACEMENT_ACTIONS)
        placements.append((li, ti))
        state, _, done, _ = env.step(action); mask = env.legal_mask()
    from .baselines import plan_metrics
    return placements, plan_metrics(env.p, placements)


@torch.no_grad()
def policy_search(env, agent, n_eval=256, taus=(0.7, 0.4, 0.2, 0.1), rng=None):
    """PURE-RL per-tile optimiser (NO greedy / local-search hybrid): best-of-N Boltzmann-sampled
    policy rollouts under a hard n_eval surrogate-eval budget. The first rollout is greedy (floor);
    the rest sample a ~ softmax(Q/tau) with tau annealed DOWN across the budget, so later samples
    sharpen around the best region. Tracks the best PREFIX (across all rollouts) by env.prev_obj —
    the FULL multi-criteria objective — and returns it. Fair vs the 256-eval search baselines."""
    from .baselines import plan_metrics
    rng = rng or np.random.default_rng(0)
    cand = env.cand_grid()
    gplan, gm = apply_policy(env, agent)                       # greedy floor
    best_plan, best_obj = gplan, gm["objective"]
    evals = len(gplan) + 1
    while evals < n_eval:
        tau = taus[min(len(taus) - 1, int(len(taus) * evals / max(1, n_eval)))]
        state = env.reset(); mask = env.legal_mask(); done = False; plan = []
        while not done and evals < n_eval:
            a = agent.act_boltzmann(state, cand, mask, tau, rng)
            state, _, done, _ = env.step(a); evals += 1
            if a != env.STOP:
                li, ti = divmod(a, N_PLACEMENT_ACTIONS); plan.append((li, ti))
                if env.prev_obj > best_obj:
                    best_obj, best_plan = env.prev_obj, list(plan)
            mask = env.legal_mask()
    return best_plan, plan_metrics(env.p, best_plan)


def local_polish(problem, plan, max_eval=2500):
    """Lazy-greedy completion seeded from `plan` (e.g. an RL rollout): fill remaining budget
    with the best marginal-objective-per-cost (loc,type), using the diminishing-returns heap
    trick (same as baselines.greedy) so it stays cheap. Bounded by max_eval surrogate evals.
    This makes RL a *learned-init local search*: RL proposes the global structure, polish
    locally perfects it. Returns the improved plan."""
    import heapq
    from .baselines import layout_from_plan
    plan = list(plan)
    used = np.zeros(problem.K, bool)
    for li, _ in plan:
        if 0 <= li < problem.K:
            used[li] = True
    cost = problem.plan_cost(plan)
    cur = problem.evaluate(layout_from_plan(problem, plan))["objective"]; evals = 1
    cheapest = min(a.cost for a in ACTIONS)
    free = [(li, ti) for li in range(problem.K) if not used[li]
            for ti in range(N_PLACEMENT_ACTIONS)]
    if not free or cost + cheapest > problem.budget:
        return plan
    res = problem.evaluate_batch([layout_from_plan(problem, plan + [c]) for c in free])
    evals += len(free)
    heap = [(-(res[k]["objective"] - cur) / ACTIONS[free[k][1]].cost, free[k][0], free[k][1], 0)
            for k in range(len(free))]
    heapq.heapify(heap); step = 0
    while heap and evals < max_eval and cost + cheapest <= problem.budget:
        neg_g, li, ti, last = heapq.heappop(heap)
        if used[li] or cost + ACTIONS[ti].cost > problem.budget:
            continue
        if last == step:                         # gain is current -> commit
            if -neg_g <= 0:
                break
            plan.append((li, ti)); used[li] = True; cost += ACTIONS[ti].cost
            cur = problem.evaluate(layout_from_plan(problem, plan))["objective"]; evals += 1
            step += 1
        else:                                    # stale -> recompute this candidate's gain
            g = (problem.evaluate(layout_from_plan(problem, plan + [(li, ti)]))["objective"] - cur) \
                / ACTIONS[ti].cost
            evals += 1
            if g > 0:
                heapq.heappush(heap, (-g, li, ti, step))
    return plan


def optimize_tile_v2(env, device="cpu", episodes=150, warm_state=None, learn_every=2,
                     eps_start=1.0, eps_end=0.02, sync_every=10, rng=None,
                     polish=False, polish_eval=2500, cost_bias=False):
    """Stronger per-tile RL optimiser. Three fixes over optimize_tile:
      1. EXPONENTIAL eps anneal all the way to ~greedy (eps_end=0.02) -> the later episodes are
         real exploitation rollouts, not random walks.
      2. Track the BEST plan over ALL steps of ALL episodes (best prefix), not just each
         episode's final layout — a good plan found mid-episode is never thrown away.
      3. Optional learned-init local search (`polish`): RL proposes structure, lazy-greedy
         perfects it locally.
    Returns (best_plan, metrics)."""
    from .baselines import plan_metrics
    rng = rng or np.random.default_rng(0)
    cand = env.cand_grid()
    agent = GeneralistAgent(device=device)
    agent.cost_bias = cost_bias
    if warm_state is not None:
        agent.online.load_state_dict(warm_state); agent.target.load_state_dict(warm_state)
    best_plan, best_obj = [], -1e9
    tick = 0
    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        eps = eps_start * (eps_end / eps_start) ** frac        # exp anneal -> eps_end
        state = env.reset(); mask = env.legal_mask(); done = False; plan = []
        while not done:
            a = agent.act(state, cand, mask, eps)
            ns, r, done, _ = env.step(a); nmask = env.legal_mask()
            if a != env.STOP:
                li, ti = divmod(a, N_PLACEMENT_ACTIONS); plan.append((li, ti))
                if env.prev_obj > best_obj:                     # best PREFIX, across all episodes
                    best_obj, best_plan = env.prev_obj, list(plan)
            agent.replay.push(state, cand, a, r, ns, done, nmask)
            tick += 1
            if tick % learn_every == 0:
                agent.learn()
            state, mask = ns, nmask
        if ep % sync_every == 0:
            agent.sync()
    gplan, gm = apply_policy(env, agent)                        # final exploitation rollout
    if gm["objective"] > best_obj:
        best_plan, best_obj = gplan, gm["objective"]
    if polish:
        best_plan = local_polish(env.p, best_plan, max_eval=polish_eval)
    return best_plan, plan_metrics(env.p, best_plan)
