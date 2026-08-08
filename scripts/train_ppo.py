#!/usr/bin/env python3
"""PPO trainer for the adaptive-measurement demon (HYAK-scale Phase 1b).

Policy class: translation-equivariant 1D circular ConvNet over per-site
classical features, plus a global GRU memory whose hidden state is broadcast
back to every site — so the agent CAN express moving patterns (sweeps), unlike
the linear CEM policy. Actions are k distinct sites per layer, sampled without
replacement via Plackett-Luce (sequential softmax), which gives exact log-probs
for PPO.

Demon legality: the policy INPUT is classical-record features only. With
--shaped, the training reward uses per-layer entropy changes (quantum side) as
shaping signal — legal, since reward is the experimenter's business; the
deployed policy still only ever sees the record. Final objective either way:
minimize final S(L/2) at fixed budget.

Local smoke test:
  python scripts/train_ppo.py --L 16 --budget 2 --T 64 --updates 5 \
      --episodes-per-update 8 --smoke

HYAK (single GPU or CPU node):
  python scripts/train_ppo.py --L 32 --budget 4 --T 128 --updates 300 \
      --episodes-per-update 64 --shaped --out results/ppo_L32b4.json
"""

import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mipt.env import AdaptiveMIPTEnv, N_FEATURES  # noqa: E402
from mipt.sim import stabilizer_matrix, region_entropy  # noqa: E402


# ---------------------------------------------------------------------------
class DemonPolicy(nn.Module):
    """Per-site logits + value head. Circular convs (PBC-equivariant) + global
    GRU memory broadcast to all sites."""

    def __init__(self, n_features: int = N_FEATURES, ch: int = 32, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, ch, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1, padding_mode="circular")
        self.gru = nn.GRUCell(ch, hidden)
        self.site_head = nn.Conv1d(ch + hidden, 1, 1)
        self.value_head = nn.Linear(ch + hidden, 1)
        self.hidden_dim = hidden

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        # obs: (B, L, F); h: (B, hidden)
        x = obs.transpose(1, 2)                      # (B, F, L)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))                    # (B, ch, L)
        pooled = x.mean(dim=2)                       # (B, ch)
        h_new = self.gru(pooled, h)                  # (B, hidden)
        hb = h_new.unsqueeze(2).expand(-1, -1, x.shape[2])
        xh = torch.cat([x, hb], dim=1)               # (B, ch+hidden, L)
        logits = self.site_head(xh).squeeze(1)       # (B, L)
        value = self.value_head(torch.cat([pooled, h_new], dim=1)).squeeze(-1)
        return logits, value, h_new

    def init_hidden(self, batch: int):
        return torch.zeros(batch, self.hidden_dim)


def sample_sites(logits: torch.Tensor, k: int):
    """Plackett-Luce top-k without replacement. logits: (L,). Returns (sites,
    logprob, entropy-ish)."""
    sites, logp = [], 0.0
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for _ in range(k):
        masked = logits.masked_fill(mask, -1e9)
        dist = torch.distributions.Categorical(logits=masked)
        s = dist.sample()
        logp = logp + dist.log_prob(s)
        mask[s] = True
        sites.append(int(s))
    return sites, logp


def logprob_of(logits: torch.Tensor, sites: list[int]):
    logp = torch.zeros((), device=logits.device)
    ent = torch.zeros((), device=logits.device)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for s in sites:
        masked = logits.masked_fill(mask, -1e9)
        dist = torch.distributions.Categorical(logits=masked)
        logp = logp + dist.log_prob(torch.tensor(s, device=logits.device))
        ent = ent + dist.entropy()
        # functional update — in-place mask edits break autograd's saved tensors
        mask = mask.scatter(0, torch.tensor(s, device=logits.device), True)
    return logp, ent


def half_cut(env: AdaptiveMIPTEnv) -> float:
    mat = stabilizer_matrix(env.sim, env.L)
    return float(region_entropy(mat, np.arange(env.L // 2), env.L))


# ---------------------------------------------------------------------------
def collect_episode(policy, L, budget, T, seed, shaped, device):
    env = AdaptiveMIPTEnv(L, budget, T, seed=seed)
    obs = env._obs()
    h = policy.init_hidden(1).to(device)
    traj = {"obs": [], "h": [], "sites": [], "logp": [], "rew": [], "val": []}
    prev_S = half_cut(env) if shaped else None
    while not env.done:
        ot = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value, h_new = policy(ot, h)
        sites, logp = sample_sites(logits[0], budget)
        traj["obs"].append(obs.copy()); traj["h"].append(h[0].cpu().numpy())
        traj["sites"].append(sites); traj["logp"].append(float(logp))
        traj["val"].append(float(value))
        obs, _ = env.step(sites)
        if shaped and not env.done:
            cur = half_cut(env)
            traj["rew"].append(prev_S - cur)  # positive if S dropped
            prev_S = cur
        else:
            traj["rew"].append(0.0)
        h = h_new
    final = env.finish()
    traj["rew"][-1] += -final["S_half"]  # terminal objective
    traj["final"] = final
    return traj


def ppo_update(policy, opt, trajs, device, epochs=4, clip=0.2, gamma=0.995,
               lam=0.95, ent_coef=0.01, vf_coef=0.5):
    # flatten with GAE per-trajectory
    all_obs, all_h, all_sites, all_logp, all_adv, all_ret = [], [], [], [], [], []
    for tr in trajs:
        rews, vals = np.array(tr["rew"]), np.array(tr["val"] + [0.0])
        adv = np.zeros(len(rews)); last = 0.0
        for t in reversed(range(len(rews))):
            delta = rews[t] + gamma * vals[t + 1] - vals[t]
            last = delta + gamma * lam * last
            adv[t] = last
        ret = adv + vals[:-1]
        all_obs += tr["obs"]; all_h += tr["h"]; all_sites += tr["sites"]
        all_logp += tr["logp"]; all_adv += adv.tolist(); all_ret += ret.tolist()
    obs = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
    h0 = torch.tensor(np.array(all_h), dtype=torch.float32, device=device)
    old_logp = torch.tensor(all_logp, dtype=torch.float32, device=device)
    adv = torch.tensor(all_adv, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = torch.tensor(all_ret, dtype=torch.float32, device=device)

    n = len(all_obs)
    for _ in range(epochs):
        logits, values, _ = policy(obs, h0)
        logps, ents = [], []
        for i in range(n):
            lp, en = logprob_of(logits[i], all_sites[i])
            logps.append(lp); ents.append(en)
        logp = torch.stack(logps); ent = torch.stack(ents)
        ratio = torch.exp(logp - old_logp)
        pg = -torch.min(ratio * adv,
                        torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        vloss = F.mse_loss(values, ret)
        loss = pg + vf_coef * vloss - ent_coef * ent.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--episodes-per-update", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shaped", action="store_true")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run, no output file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    policy = DemonPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    history = []
    t0 = time.time()
    for u in range(args.updates):
        trajs = [collect_episode(policy, args.L, args.budget, args.T,
                                 seed=args.seed * 10_007 + u * 1000 + e,
                                 shaped=args.shaped, device=device)
                 for e in range(args.episodes_per_update)]
        loss = ppo_update(policy, opt, trajs, device)
        S = np.mean([tr["final"]["S_half"] for tr in trajs])
        H = np.mean([tr["final"]["record_entropy_exact"] for tr in trajs])
        history.append({"update": u, "S_half": float(S),
                        "record_H": float(H), "loss": loss})
        print(f"upd {u:3d}  S_half = {S:.3f}  record H = {H:.1f} bits  "
              f"loss = {loss:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(policy.state_dict(), args.out.replace(".json", ".pt"))
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "history": history}, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
