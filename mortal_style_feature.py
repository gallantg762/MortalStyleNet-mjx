import math
import numpy as np
from mjx.const import EventType
from mjx import utils

class MortalStyleFeature:

    N_CHANNELS = 506

    @classmethod
    def produce(cls, obs) -> np.ndarray:
        out = np.zeros((cls.N_CHANNELS, 34), dtype=np.float32)
        ctx = cls._precompute(obs)
        idx = 0

        idx = cls._current_hand(out, idx, obs)
        idx = cls._target_tile(out, idx, obs)
        idx = cls._under_riichis(out, idx, ctx)
        idx = cls._riichi_declared_tiles(out, idx, ctx)
        idx = cls._dealer(out, idx, obs)
        idx = cls._doras(out, idx, obs)
        idx = cls._shanten(out, idx, obs)
        idx = cls._effective_discards(out, idx, obs)
        idx = cls._effective_draws(out, idx, obs)
        idx = cls._ignored_tiles(out, idx, ctx)
        idx = cls._discarded_tiles_detail(out, idx, ctx)
        idx = cls._discarded_decay(out, idx, ctx)
        idx = cls._tedashi_decay(out, idx, ctx)
        idx = cls._opened_tiles(out, idx, ctx)
        idx = cls._tiles_seen(out, idx, ctx)
        idx = cls._legal_action_flags(out, idx, obs)
        idx = cls._scores(out, idx, obs, ctx)
        idx = cls._rankings(out, idx, obs)
        idx = cls._round(out, idx, obs)
        idx = cls._honba(out, idx, obs)
        idx = cls._kyotaku(out, idx, obs)
        idx = cls._bakaze_jikaze(out, idx, obs)
        idx = cls._tiles_left(out, idx, ctx)

        assert idx == cls.N_CHANNELS
        return out

    @staticmethod
    def _precompute(obs):
        who = obs.who()
        dora_set = set(obs.doras())
        kawa = [[] for _ in range(4)]
        opened = [[0] * 34 for _ in range(4)]
        is_red_opened = [[False] * 3 for _ in range(4)]
        riichi_declared = [False] * 4
        riichi_pending = {}
        ignored = [set() for _ in range(4)]
        draw_count = 0

        for e in obs.events():
            etype = e.type()

            if etype == EventType.DRAW:
                draw_count += 1

            elif etype == EventType.RIICHI:
                riichi_declared[e.who()] = True
                riichi_pending[e.who()] = None

            elif etype in (EventType.DISCARD, EventType.TSUMOGIRI):
                p = e.who()
                tile = e.tile()
                tt = tile.type()
                is_tedashi = (etype == EventType.DISCARD)
                kawa[p].append((tt, is_tedashi, riichi_declared[p], tile.is_red()))
                if p in riichi_pending and riichi_pending[p] is None:
                    riichi_pending[p] = tile
                ignored[p].add(tt)
                if is_tedashi:
                    ignored[p] = {tt}

            elif etype in (EventType.CHI, EventType.PON,
                           EventType.CLOSED_KAN, EventType.OPEN_KAN):
                p = e.who()
                for t in e.open().tiles():
                    opened[p][t.type()] += 1
                    if t.is_red():
                        is_red_opened[p][t.type() // 9] = True

            elif etype == EventType.ADDED_KAN:
                p = e.who()
                t = e.open().last_tile()
                opened[p][t.type()] += 1
                if t.is_red():
                    is_red_opened[p][t.type() // 9] = True

        tiles_seen = [0] * 34
        for p in range(4):
            for tt, _, _, _ in kawa[p]:
                tiles_seen[tt] += 1
            for tt in range(34):
                tiles_seen[tt] += opened[p][tt]
        hand = obs.curr_hand()
        for tt, cnt in enumerate(hand.closed_tile_types()):
            tiles_seen[tt] += cnt
        for meld in hand.opens():
            for t in meld.tiles():
                tiles_seen[t.type()] += 1

        return {
            "who": who,
            "dora_set": dora_set,
            "kawa": kawa,
            "opened": opened,
            "is_red_opened": is_red_opened,
            "riichi_declared": riichi_declared,
            "riichi_pending": riichi_pending,
            "ignored": ignored,
            "tiles_seen": tiles_seen,
            "draw_count": draw_count,
        }

    @staticmethod
    def _current_hand(out, idx, obs):
        # 7ch
        hand = obs.curr_hand()
        for tt, cnt in enumerate(hand.closed_tile_types()):
            for i in range(min(cnt, 4)):
                out[idx + i, tt] = 1.0
        for tile in hand.closed_tiles():
            if tile.is_red():
                out[idx + 4 + (tile.type() // 9), :] = 1.0
        return idx + 7

    @staticmethod
    def _target_tile(out, idx, obs):
        # 2ch
        events = obs.events()
        if not events:
            return idx + 2
        last = events[-1]
        etype = last.type()
        if etype in (EventType.DISCARD, EventType.TSUMOGIRI):
            tile = last.tile()
            out[idx, tile.type()] = 1.0
            if tile.is_red():
                out[idx + 1, :] = 1.0
        elif etype == EventType.DRAW:
            draws = obs.draws()
            if draws:
                out[idx, draws[-1].type()] = 1.0
        elif etype == EventType.ADDED_KAN:
            tile = last.open().last_tile()
            out[idx, tile.type()] = 1.0
            if tile.is_red():
                out[idx + 1, :] = 1.0
        return idx + 2

    @staticmethod
    def _under_riichis(out, idx, ctx):
        # 4ch
        who = ctx["who"]
        for p, declared in enumerate(ctx["riichi_declared"]):
            if declared:
                q = (p - who + 4) % 4
                out[idx + q, :] = 1.0
        return idx + 4

    @staticmethod
    def _riichi_declared_tiles(out, idx, ctx):
        # 6ch
        who = ctx["who"]
        dora_set = ctx["dora_set"]
        for p, tile in ctx["riichi_pending"].items():
            if p == who or tile is None:
                continue
            q = (p - who - 1 + 4) % 4
            if q >= 3:
                continue
            tt = tile.type()
            out[idx + q * 3, tt] = 1.0
            if tile.is_red():
                out[idx + q * 3 + 1, :] = 1.0
            if tt in dora_set:
                out[idx + q * 3 + 2, :] = 1.0
        return idx + 6

    @staticmethod
    def _dealer(out, idx, obs):
        q = (obs.dealer() - obs.who() + 4) % 4
        out[idx + q, :] = 1.0
        return idx + 4

    @staticmethod
    def _doras(out, idx, obs):
        cnt = [0] * 34
        for d in obs.doras():
            cnt[d] += 1
        for tt in range(34):
            for i in range(min(cnt[tt], 4)):
                out[idx + i, tt] = 1.0
        return idx + 4

    @staticmethod
    def _shanten(out, idx, obs):
        n = obs.curr_hand().shanten_number()
        for i in range(min(n + 1, 7)):
            out[idx + i, :] = 1.0
        return idx + 7

    @staticmethod
    def _effective_discards(out, idx, obs):
        for tt in obs.curr_hand().effective_discard_types():
            out[idx, tt] = 1.0
        return idx + 1

    @staticmethod
    def _effective_draws(out, idx, obs):
        for tt in obs.curr_hand().effective_draw_types():
            out[idx, tt] = 1.0
        return idx + 1

    @staticmethod
    def _ignored_tiles(out, idx, ctx):
        who = ctx["who"]
        for p, tiles in enumerate(ctx["ignored"]):
            q = (p - who + 4) % 4
            for tt in tiles:
                out[idx + q, tt] = 1.0
        return idx + 4

    @staticmethod
    def _encode_kawa_block(out, idx, items, n_items):
        for i in range(n_items):
            if i < len(items):
                tt, is_tedashi, is_riichi, is_red = items[i]
                out[idx, tt] = 1.0
                if is_red:
                    out[idx + 1, :] = 1.0
                if is_tedashi:
                    out[idx + 2, tt] = 1.0
                if is_riichi:
                    out[idx + 3, tt] = 1.0
            idx += 4
        return idx

    @classmethod
    def _discarded_tiles_detail(cls, out, idx, ctx):
        who = ctx["who"]
        kawa = ctx["kawa"]
        idx = cls._encode_kawa_block(out, idx, kawa[who][:6],   6)
        idx = cls._encode_kawa_block(out, idx, kawa[who][-18:], 18)

        for i in range(1, 4):
            p = (who + i) % 4
            idx = cls._encode_kawa_block(out, idx, kawa[p][:6],   6)
            idx = cls._encode_kawa_block(out, idx, kawa[p][-18:], 18)
        return idx

    @staticmethod
    def _discarded_decay(out, idx, ctx):
        who = ctx["who"]
        kawa = ctx["kawa"]
        max_len = max((len(k) for k in kawa), default=1)
        for p in range(4):
            q = (p - who + 4) % 4
            for turn, (tt, _, _, _) in enumerate(kawa[p]):
                out[idx + q, tt] = math.exp(-0.2 * (max_len - 1 - turn))
        return idx + 4

    @staticmethod
    def _tedashi_decay(out, idx, ctx):
        who = ctx["who"]
        kawa = ctx["kawa"]
        max_len = max((len(k) for k in kawa), default=1)
        for i in range(1, 4):
            p = (who + i) % 4
            for turn, (tt, is_tedashi, _, _) in enumerate(kawa[p]):
                if is_tedashi:
                    out[idx + i - 1, tt] = math.exp(-0.2 * (max_len - 1 - turn))
        return idx + 3

    @staticmethod
    def _opened_tiles(out, idx, ctx):
        who = ctx["who"]
        for p in range(4):
            q = (p - who + 4) % 4
            base = idx + q * 7
            for tt in range(34):
                for i in range(min(ctx["opened"][p][tt], 4)):
                    out[base + i, tt] = 1.0
            for suit in range(3):
                if ctx["is_red_opened"][p][suit]:
                    out[base + 4 + suit, :] = 1.0
        return idx + 28

    @staticmethod
    def _tiles_seen(out, idx, ctx):
        ts = ctx["tiles_seen"]
        for tt in range(34):
            out[idx, tt] = min(ts[tt], 4) / 4.0
        return idx + 1

    @staticmethod
    def _legal_action_flags(out, idx, obs):
        mask = obs.action_mask()
        if mask[177]:              out[idx, :]     = 1.0
        if any(mask[74:104]):      out[idx + 1, :] = 1.0
        if any(mask[104:141]):     out[idx + 2, :] = 1.0
        if any(mask[141:175]):     out[idx + 3, :] = 1.0
        if mask[175] or mask[176]: out[idx + 4, :] = 1.0
        if mask[178]:              out[idx + 5, :] = 1.0
        return idx + 6

    @staticmethod
    def _scores(out, idx, obs, ctx):
        who = ctx["who"]
        for i, score in enumerate(obs.tens()):
            q = (i - who + 4) % 4
            out[idx + q * 2,     :] = min(max(score, 0), 100_000) / 100_000.0
            out[idx + q * 2 + 1, :] = min(max(score, 0),  30_000) /  30_000.0
        return idx + 8

    @staticmethod
    def _rankings(out, idx, obs):
        who = obs.who()
        for p, r in enumerate(utils.rankings(obs.tens())):
            q = (p - who + 4) % 4
            for i in range(r):
                out[idx + q * 3 + i, :] = 1.0
        return idx + 12

    @staticmethod
    def _round(out, idx, obs):
        for i in range(min(obs.round(), 7)):
            out[idx + i, :] = 1.0
        return idx + 7

    @staticmethod
    def _honba(out, idx, obs):
        for i in range(min(obs.honba(), 5)):
            out[idx + i, :] = 1.0
        return idx + 5

    @staticmethod
    def _kyotaku(out, idx, obs):
        for i in range(min(obs.kyotaku(), 5)):
            out[idx + i, :] = 1.0
        return idx + 5

    @staticmethod
    def _bakaze_jikaze(out, idx, obs):
        bakaze = obs.round() // 4
        jikaze = (obs.who() - obs.dealer() + 4) % 4
        out[idx,     27 + bakaze] = 1.0
        out[idx + 1, 27 + jikaze] = 1.0
        return idx + 2

    @staticmethod
    def _tiles_left(out, idx, ctx):
        tiles_left = max(0, 70 - ctx["draw_count"])
        out[idx, :] = tiles_left / 69.0
        return idx + 1