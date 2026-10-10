"""Entity-token policy: what is on the board, what is in hand, what the
opponent has shown -> wait or play (which hand card, which tile), plus value.

Inputs are the columns `sim.replay_rows` writes. Every board object is a token
(unit id embedding plus its features), so the network sees *what* each unit
is, not only where something is - the plane observation in `sim.env` counted
units per cell and could not tell a Skeleton from a Golem.

Heads:
* gate + card: a pointer over the four hand cards plus a wait logit.
* tile: 18 x 32 logits conditioned on the chosen card's embedding.
* value: the 10 s tower outcome the row was labelled with, and the win.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_W, GRID_H = 18, 32
TILES = GRID_W * GRID_H


class TokenPolicy(nn.Module):
    def __init__(self, n_cards: int, n_units: int, d: int = 192, layers: int = 4,
                 heads: int = 6, unit_features: int = 7):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards, d, padding_idx=0)
        self.unit_emb = nn.Embedding(n_units, d, padding_idx=0)
        self.unit_in = nn.Linear(unit_features, d)
        self.state_in = nn.Linear(6 + 4, d)            # towers + scalars
        self.type_emb = nn.Embedding(6, d)             # cls, unit, state, hand, next, opp
        self.slot_emb = nn.Embedding(4, d)
        self.age_in = nn.Linear(1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.wait_head = nn.Linear(d, 1)
        self.card_query = nn.Linear(d, d)
        self.tile_head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(),
                                       nn.Linear(2 * d, TILES))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def encode(self, b):
        n = b["unit_ids"].shape[0]
        dev = b["unit_ids"].device
        t = lambda i: self.type_emb.weight[i]
        units = (self.unit_emb(b["unit_ids"]) + self.unit_in(b["unit_feats"].float())
                 + t(1))
        state = self.state_in(torch.cat([b["towers"], b["scalars"]], -1))[:, None] + t(2)
        hand = (self.card_emb(b["hand"]) + self.slot_emb.weight[None] + t(3))
        nxt = self.card_emb(b["next"])[:, None] + t(4)
        seen = self.card_emb(b["opp_seen"]) + t(5)
        last = (self.card_emb(b["opp_last"]) + self.age_in(b["opp_last_age"].float()[..., None])
                + t(5))
        cls = self.cls.expand(n, -1, -1) + t(0)
        x = torch.cat([cls, state, hand, nxt, units, seen, last], 1)
        pad = torch.cat([
            torch.zeros(n, 2, dtype=torch.bool, device=dev),
            b["hand"] == 0, b["next"][:, None] == 0, b["unit_ids"] == 0,
            b["opp_seen"] == 0, b["opp_last"] == 0], 1)
        pad[:, 0] = False
        h = self.norm(self.encoder(x, src_key_padding_mask=pad))
        return h[:, 0], h[:, 2:6]

    def forward(self, b, card_ids=None):
        """Returns gate logits [N, 5] (wait + 4 hand slots), tile logits [N, TILES]
        for `card_ids` (default: the slot with the highest logit), and value [N, 2]."""
        cls, hand_h = self.encode(b)
        q = self.card_query(cls)
        slot_logits = torch.einsum("nd,nkd->nk", q, hand_h) / hand_h.shape[-1] ** 0.5
        slot_logits = slot_logits.masked_fill(b["hand"] == 0, -1e4)
        gate = torch.cat([self.wait_head(cls), slot_logits], 1)
        if card_ids is None:
            slot = slot_logits.argmax(1)
            card_ids = b["hand"].gather(1, slot[:, None])[:, 0]
        tile = self.tile_head(torch.cat([cls, self.card_emb(card_ids)], -1))
        return gate, tile, self.value_head(cls)


def target_slot(hand: torch.Tensor, card: torch.Tensor) -> torch.Tensor:
    """Gate target: 0 for wait, 1 + hand slot of the played card, -1 if the
    played card is not in the recorded hand (row skipped for the gate)."""
    match = (hand == card[:, None]) & (card[:, None] > 0)
    slot = match.float().argmax(1) + 1
    has = match.any(1)
    out = torch.where(card == 0, torch.zeros_like(slot), slot)
    return torch.where((card > 0) & ~has, torch.full_like(slot, -1), out)


def losses(model, b, beta: float = 0.0, max_weight: float = 5.0):
    """Imitation losses, optionally outcome-weighted.

    beta == 0: plain imitation. beta > 0: each row's weight is
    exp(advantage / beta) clipped to `max_weight`, where the advantage is the
    row's 10 s tower outcome minus the value head's prediction for that state
    (advantage-weighted regression). The value head learns the outcome either
    way, so the two runs differ only in the weights.
    """
    card = b["act_card"].long()
    gate_t = target_slot(b["hand"], card)
    gate, tile, value = model(b, card_ids=card.clamp(min=0))
    v_out = value[:, 0]
    v_won = value[:, 1]
    outcome = b["outcome"].float()
    if beta > 0:
        adv = (outcome - v_out.detach())
        w = torch.exp(adv / beta).clamp(max=max_weight)
    else:
        w = torch.ones_like(outcome)
    ok = gate_t >= 0
    gate_loss = (F.cross_entropy(gate[ok], gate_t[ok], reduction="none") * w[ok]).sum() / w[ok].sum().clamp(min=1e-6)
    plays = card > 0
    tile_t = (b["act_y"].long() * GRID_W + b["act_x"].long())
    tile_loss = ((F.cross_entropy(tile[plays], tile_t[plays], reduction="none") * w[plays]).sum()
                 / w[plays].sum().clamp(min=1e-6)) if plays.any() else gate_loss * 0
    value_loss = F.mse_loss(v_out, outcome) + F.binary_cross_entropy_with_logits(v_won, b["won"].float())
    with torch.no_grad():
        stats = {
            "gate_acc": (gate[ok].argmax(1) == gate_t[ok]).float().mean().item() if ok.any() else 0.0,
            "card_acc": ((gate[ok & plays][:, 1:].argmax(1) + 1) == gate_t[ok & plays]).float().mean().item()
            if (ok & plays).any() else 0.0,
            "tile_acc": (tile[plays].argmax(1) == tile_t[plays]).float().mean().item() if plays.any() else 0.0,
            "tile_near": _near(tile[plays].argmax(1), tile_t[plays]) if plays.any() else 0.0,
            "mean_w": w.mean().item(),
        }
    return gate_loss + tile_loss + 0.5 * value_loss, stats


def _near(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Share of predicted tiles within one tile (Chebyshev) of the pro's."""
    px, py = pred % GRID_W, pred // GRID_W
    tx, ty = true % GRID_W, true // GRID_W
    return (((px - tx).abs() <= 1) & ((py - ty).abs() <= 1)).float().mean().item()
