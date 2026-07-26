"""Dueling Double DQN for micro-update planning (the paper's method).

CNN encodes the spatial state, a shared MLP fuses the budget/progress scalars, and a
dueling head splits Q = V(s) + (A(s,a) − mean_a A). Double-DQN target (online net picks
the argmax, target net values it) curbs overestimation. Action masking zeroes illegal
moves. Matches 微更新研究方案说明.md §3.
"""
from __future__ import annotations

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):
    def __init__(self, in_ch: int, n_scalar: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.scalar = nn.Sequential(nn.Linear(n_scalar, 64), nn.ReLU(inplace=True))
        self.shared = nn.Sequential(nn.Linear(64 * 16 + 64, hidden), nn.ReLU(inplace=True))
        self.V = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.A = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, n_actions))

    def forward(self, raster, scalar):
        h = self.conv(raster).flatten(1)
        h = self.shared(torch.cat([h, self.scalar(scalar)], dim=1))
        v, a = self.V(h), self.A(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class SpatialDuelingQNet(nn.Module):
    """Fully-conv dueling Q-net for spatial placement. Outputs a per-(type,cell) advantage
    map, gathers it at the fixed candidate cells, and adds a global V(s) and a STOP head.
    Preserves spatial detail that a global-pooled head loses → far easier to value 'place
    a tree at *this* hot cell'. Action index = loc*n_types + type (loc-major), STOP last —
    matching MicroUpdateEnv. forward signature identical to DuelingQNet (drop-in)."""

    def __init__(self, in_ch, n_scalar, n_types, cand_coords, hidden=64, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.register_buffer("cand", torch.as_tensor(cand_coords, dtype=torch.long))  # (K,2) in state grid
        self.K = int(self.cand.shape[0]); self.n_types = n_types
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.scalar = nn.Linear(n_scalar, hidden)
        self.adv = nn.Conv2d(hidden, n_types, 1)
        self.val = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.stop = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, raster, scalar):
        f = self.enc(raster)
        f = f + self.scalar(scalar).unsqueeze(-1).unsqueeze(-1)
        adv_map = self.adv(f)                                   # (B,n_types,Hs,Ws)
        rr, cc = self.cand[:, 0], self.cand[:, 1]
        adv_cand = adv_map[:, :, rr, cc]                        # (B,n_types,K)
        B = f.shape[0]
        adv_cand = adv_cand.permute(0, 2, 1).reshape(B, self.K * self.n_types)  # loc-major
        pooled = f.mean(dim=(2, 3))
        adv = torch.cat([adv_cand, self.stop(pooled)], dim=1)  # (B, K*n_types+1)
        if not self.dueling:                       # ablation: raw Q head, no V/A split
            return adv
        return self.val(pooled) + (adv - adv.mean(dim=1, keepdim=True))


class Replay:
    def __init__(self, cap=6000):
        self.buf = deque(maxlen=cap)

    def push(self, *t):
        self.buf.append(t)

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, in_ch, n_scalar, n_actions, device="cpu", gamma=0.97, lr=5e-4,
                 cand_coords=None, n_types=None, dueling=True, double=True):
        self.device = device
        self.n_actions = n_actions
        self.double = double          # ablation A2 switch

        def make():
            if cand_coords is not None:
                return SpatialDuelingQNet(in_ch, n_scalar, n_types, cand_coords, dueling=dueling)
            return DuelingQNet(in_ch, n_scalar, n_actions)
        self.online = make().to(device)
        self.target = make().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = Replay()

    def _tensify(self, states):
        r = torch.from_numpy(np.stack([s["raster"] for s in states]).astype(np.float32)).to(self.device)
        s = torch.from_numpy(np.stack([s["scalar"] for s in states])).to(self.device)
        return r, s

    @torch.no_grad()
    def act(self, state, mask, eps):
        if random.random() < eps:
            legal = np.flatnonzero(mask)
            return int(random.choice(legal))
        r, s = self._tensify([state])
        q = self.online(r, s).squeeze(0).cpu().numpy()
        q[~mask] = -1e9
        return int(q.argmax())

    def learn(self, batch_size=64):
        if len(self.replay) < batch_size * 4:
            return None
        batch = self.replay.sample(batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        r, s = self._tensify(states)
        nr, ns = self._tensify(next_states)
        a = torch.tensor(actions, device=self.device).long()
        rew = torch.tensor(rewards, device=self.device).float()
        done = torch.tensor(dones, device=self.device).float()
        nmask = torch.from_numpy(np.stack(next_masks)).to(self.device)
        q = self.online(r, s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:                       # Double DQN: online picks, target evaluates
                nq_online = self.online(nr, ns)
                nq_online[~nmask] = -1e9
                best = nq_online.argmax(dim=1, keepdim=True)
                nq_target = self.target(nr, ns).gather(1, best).squeeze(1)
            else:                                 # ablation: vanilla DQN target (max over target net)
                nq = self.target(nr, ns)
                nq[~nmask] = -1e9
                nq_target = nq.max(dim=1).values
            y = rew + self.gamma * nq_target * (1 - done)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync(self):
        self.target.load_state_dict(self.online.state_dict())


def train_dqn(env, agent, episodes=300, eps_start=1.0, eps_end=0.05, eps_decay=0.985,
              sync_every=10, learn_every=2, log=None):
    eps = eps_start
    history = []
    tick = 0
    for ep in range(episodes):
        state = env.reset()
        mask = env.legal_mask()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(state, mask, eps)
            next_state, reward, done, _ = env.step(action)
            next_mask = env.legal_mask()
            agent.replay.push(state, action, reward, next_state, done, next_mask)
            tick += 1
            if tick % learn_every == 0:
                agent.learn()
            state, mask = next_state, next_mask
            ep_reward += reward
        eps = max(eps_end, eps * eps_decay)
        if (ep + 1) % sync_every == 0:
            agent.sync()
        history.append(ep_reward)
        if log and (ep + 1) % log == 0:
            obj = env.p.evaluate(env.layout)["objective"]
            print(f"    ep {ep+1:3}/{episodes} eps={eps:.2f} ep_R={ep_reward:+.3f} "
                  f"final_obj={obj:.3f} cost={env.cost:.0f}", flush=True)
    return history


@torch.no_grad()
def rollout_plan(env, agent):
    """Greedy (eps=0) rollout of the learned policy -> (placements, layout, metrics)."""
    from .action_space import N_PLACEMENT_ACTIONS
    state = env.reset()
    mask = env.legal_mask()
    placements = []
    done = False
    while not done:
        action = agent.act(state, mask, 0.0)
        if action == env.STOP:
            break
        li, ti = divmod(action, N_PLACEMENT_ACTIONS)
        placements.append((li, ti))
        state, reward, done, _ = env.step(action)
        mask = env.legal_mask()
    metrics = env.p.evaluate(env.layout)
    metrics = {k: v for k, v in metrics.items() if k != "dutci_field"}
    return placements, env.layout, {**metrics, "cost": env.cost, "n_actions": len(placements)}
