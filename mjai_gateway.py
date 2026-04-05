# Gateway for converting between MJAI protocol events and mjx observations/actions.

import copy
import json
import torch
from typing import Any

import mjx
from mjx import Observation
from mjx.open import Open
from mjx.event import Event
from mjx.const import TileType, EventType, ActionType
from mjx.tile import Tile

def to_mjx_tile(tile_str: str, ignore_aka: bool = False) -> int:
    if tile_str == "1m": return 0
    if tile_str == "2m": return 4
    if tile_str == "3m": return 8
    if tile_str == "4m": return 12
    if tile_str == "5mr": return 16
    if tile_str == "5m": return (16 if ignore_aka else 17)
    if tile_str == "6m": return 20
    if tile_str == "7m": return 24
    if tile_str == "8m": return 28
    if tile_str == "9m": return 32
    if tile_str == "1p": return 36
    if tile_str == "2p": return 40
    if tile_str == "3p": return 44
    if tile_str == "4p": return 48
    if tile_str == "5pr": return 52
    if tile_str == "5p": return (52 if ignore_aka else 53)
    if tile_str == "6p": return 56
    if tile_str == "7p": return 60
    if tile_str == "8p": return 64
    if tile_str == "9p": return 68
    if tile_str == "1s": return 72
    if tile_str == "2s": return 76
    if tile_str == "3s": return 80
    if tile_str == "4s": return 84
    if tile_str == "5sr": return 88
    if tile_str == "5s": return (88 if ignore_aka else 89)
    if tile_str == "6s": return 92
    if tile_str == "7s": return 96
    if tile_str == "8s": return 100
    if tile_str == "9s": return 104
    if tile_str == "E": return 108
    if tile_str == "S": return 112
    if tile_str == "W": return 116
    if tile_str == "N": return 120
    if tile_str == "P": return 124
    if tile_str == "F": return 128
    if tile_str == "C": return 132

def to_mjai_tile(tile_id: int) -> str:
    tile = Tile(tile_id)
    tile_type = tile.type()
    if tile_type == TileType.M1: return "1m"
    if tile_type == TileType.M2: return "2m"
    if tile_type == TileType.M3: return "3m"
    if tile_type == TileType.M4: return "4m"
    if tile_type == TileType.M5: return "5mr" if tile.is_red() else "5m"
    if tile_type == TileType.M6: return "6m"
    if tile_type == TileType.M7: return "7m"
    if tile_type == TileType.M8: return "8m"
    if tile_type == TileType.M9: return "9m"
    if tile_type == TileType.P1: return "1p"
    if tile_type == TileType.P2: return "2p"
    if tile_type == TileType.P3: return "3p"
    if tile_type == TileType.P4: return "4p"
    if tile_type == TileType.P5: return "5pr" if tile.is_red() else "5p"
    if tile_type == TileType.P6: return "6p"
    if tile_type == TileType.P7: return "7p"
    if tile_type == TileType.P8: return "8p"
    if tile_type == TileType.P9: return "9p"
    if tile_type == TileType.S1: return "1s"
    if tile_type == TileType.S2: return "2s"
    if tile_type == TileType.S3: return "3s"
    if tile_type == TileType.S4: return "4s"
    if tile_type == TileType.S5: return "5sr" if tile.is_red() else "5s"
    if tile_type == TileType.S6: return "6s"
    if tile_type == TileType.S7: return "7s"
    if tile_type == TileType.S8: return "8s"
    if tile_type == TileType.S9: return "9s"
    if tile_type == TileType.EW: return "E"
    if tile_type == TileType.SW: return "S"
    if tile_type == TileType.WW: return "W"
    if tile_type == TileType.NW: return "N"
    if tile_type == TileType.WD: return "P"
    if tile_type == TileType.GD: return "F"
    if tile_type == TileType.RD: return "C"
    raise ValueError(f"Invalid tile_id: {tile_id}")

def json_dumps(json_data):
    return json.dumps(json_data, separators=(",", ":"))

def _last_tile_from_events(events: list[dict]) -> int:
    for ev in reversed(events):
        if "tile" in ev:
            return ev["tile"]
    raise KeyError("No event with 'tile' key found in events")

class OpenCodeGen:
    def __init__(self):
        pass

    @staticmethod
    def from_mjai_kan(ev: dict[str, Any], obs: dict[str, Any]) -> tuple[int, list[int]]:
        pai = ev["consumed"][0]
        base = to_mjx_tile(pai, ignore_aka=True) // 4
        if "target" in ev:
            rel_pos = (ev["target"] - ev["actor"] + 4) % 4
            called = 3
        else:
            rel_pos = 0
            called = 0
        value = rel_pos
        value += ((base * 4 + called) << 8)
        new_op = Open(value)
        consume_tiles_from_hand = [t.id() for t in new_op.tiles_from_hand()]
        return value, consume_tiles_from_hand

    @staticmethod
    def from_mjai_kakan(ev: dict[str, Any], obs: dict[str, Any]) -> tuple[int, list[int], int, int]:
        pai_base = to_mjx_tile(ev['pai'], ignore_aka=True) // 4
        op = None
        if ev["actor"] == obs["who"]:
            for open_code in obs["privateObservation"]["currHand"]["opens"]:
                op = Open(open_code)
                if pai_base == op.tiles()[0].id() // 4:
                    break
        else:
            for event in obs["publicObservation"]["events"]:
                if event.get("type") == "EVENT_TYPE_PON":
                    op = Open(event["open"])
                    if pai_base == op.tiles()[0].id() // 4:
                        break
        assert op is not None
        value = ((0xffff ^ (1 << 3)) & op.bit) | (1 << 4)
        new_op = Open(value)
        return value, [t.id() for t in new_op.tiles_from_hand()], new_op.last_tile().id(), op.bit

    @staticmethod
    def from_mjai_pon(ev: dict[str, Any], obs: dict[str, Any]) -> tuple[int, list[int]]:
        pai = ev["pai"]
        base = to_mjx_tile(pai, ignore_aka=True)
        base_lowest = base // 4
        called_tile = _last_tile_from_events(obs["publicObservation"]["events"]) - base
        called = 0 if pai.endswith("r") else 2
        if ev["actor"] == obs["who"]:
            consume_tiles_from_hand = [
                t for t in obs["privateObservation"]["currHand"]["closedTiles"]
                if to_mjx_tile(to_mjai_tile(t), ignore_aka=True) == base
            ][:2]
            available_codes = [0, 1, 2, 3]
            for t in consume_tiles_from_hand:
                available_codes.remove(t - base)
            available_codes.remove(called_tile)
            not_pon = available_codes[0]
        else:
            not_consume_tiles_from_hand = [
                t for t in obs["privateObservation"]["currHand"]["closedTiles"]
                if to_mjx_tile(to_mjai_tile(t), ignore_aka=True) == base
            ] + [
                t for t in obs["publicObservation"]["doraIndicators"]
                if to_mjx_tile(to_mjai_tile(t), ignore_aka=True) == base
            ]
            candidates = [i + to_mjx_tile(ev['pai'], ignore_aka=True) for i in range(4)]
            candidates.remove(called_tile + base)
            for t in not_consume_tiles_from_hand:
                candidates.remove(t)
            for e_raw in obs["publicObservation"]["events"][:-1]:
                if any(k in e_raw for k in ['tile', 'open']):
                    e = Event(e_raw)
                    if e.type() == EventType.EVENT_TYPE_DRAW:
                        t = e.tile().id()
                        if t in candidates: candidates.remove(t)
                    elif e.type() in [EventType.EVENT_TYPE_CHI, EventType.EVENT_TYPE_PON]:
                        for t in [tile.id() for tile in e.open().tiles()]:
                            if t in candidates: candidates.remove(t)
            consume_tiles_from_hand = candidates[:2]
            available_codes = [0, 1, 2, 3]
            for t in consume_tiles_from_hand:
                available_codes.remove(t - base)
            available_codes.remove(called_tile)
            not_pon = available_codes[0]
        rel_pos = (ev["target"] - ev["actor"] + 4) % 4
        value = rel_pos + (1 << 3) + (not_pon << 5) + ((base_lowest * 3 + called) << 9)
        return value, consume_tiles_from_hand

    @staticmethod
    def from_mjai_chi(ev: dict[str, Any], obs: dict[str, Any]):
        pai, consumed = ev['pai'], ev['consumed']
        rel_pos = (ev["target"] - ev["actor"] + 4) % 4
        base = min([to_mjx_tile(p, ignore_aka=True) for p in [pai] + consumed])
        tile_from_last_event = _last_tile_from_events(obs["publicObservation"]["events"])
        called = (to_mjx_tile(pai, ignore_aka=True) - base) // 4
        if ev["actor"] == obs["who"]:
            consume0 = [t for t in obs["privateObservation"]["currHand"]["closedTiles"] if (t // 4) == (to_mjx_tile(consumed[0], ignore_aka=True) // 4)][0]
            consume1 = [t for t in obs["privateObservation"]["currHand"]["closedTiles"] if (t // 4) == (to_mjx_tile(consumed[1], ignore_aka=True) // 4)][0]
        else:
            res_consumes = []
            for c_pai in consumed:
                if c_pai.endswith('r'):
                    res_consumes.append(to_mjx_tile(c_pai))
                else:
                    c_base = to_mjx_tile(c_pai)
                    c_cands = [i + c_base for i in range(4)]
                    if c_pai.startswith('5'): c_cands.remove(c_base)
                    for t in obs["privateObservation"]["initHand"]["closedTiles"]:
                        if t in c_cands: c_cands.remove(t)
                    for e_raw in obs["publicObservation"]["events"]:
                        if any(k in e_raw for k in ['tile', 'open']):
                            e = Event(e_raw)
                            if e.type() == EventType.EVENT_TYPE_DRAW:
                                if e.tile().id() in c_cands: c_cands.remove(e.tile().id())
                            elif e.type() in [EventType.EVENT_TYPE_CHI, EventType.EVENT_TYPE_PON]:
                                for t in [tile.id() for tile in e.open().tiles()]:
                                    if t in c_cands: c_cands.remove(t)
                    res_consumes.append(c_cands[0])
            consume0, consume1 = res_consumes
        c_base = to_mjx_tile(ev['pai'])
        if called == 0:
            t0, t1, t2 = tile_from_last_event - c_base, consume0 - to_mjx_tile(consumed[0], ignore_aka=True), consume1 - to_mjx_tile(consumed[1], ignore_aka=True)
        elif called == 1:
            t1, t0, t2 = tile_from_last_event - c_base, consume0 - to_mjx_tile(consumed[0], ignore_aka=True), consume1 - to_mjx_tile(consumed[1], ignore_aka=True)
        else:
            t2, t0, t1 = tile_from_last_event - c_base, consume0 - to_mjx_tile(consumed[0], ignore_aka=True), consume1 - to_mjx_tile(consumed[1], ignore_aka=True)
        b_low = base // 4
        value = (((b_low // 9) * 7 + b_low % 9) * 3 + called) << 10
        value += (t2 << 7) + (t1 << 5) + (t0 << 3) + (1 << 2) + rel_pos
        return value, [consume0, consume1]

class MjxGateway:
    def __init__(self, actor_id, mjx_bot):
        self.actor_id = actor_id
        self.mjx_bot = mjx_bot
        self.base_obs = {}
        self.hai_offset = {}
        self.last_discard_actor = 0

    def get_obs_open(self) -> list[int]:
        if not self.base_obs: raise ValueError("Kyoku not started.")
        return self.base_obs["privateObservation"]["currHand"]["opens"]

    def get_obs_hand(self) -> list[int]:
        if not self.base_obs: raise ValueError("Kyoku not started.")
        return self.base_obs["privateObservation"]["currHand"]["closedTiles"]

    def get_obs(self) -> dict[str, Any]:
        return self.base_obs

    def set_obs_offset(self, base_obs, hai_offset) -> None:
        self.base_obs = base_obs
        self.hai_offset = hai_offset

    def get_legal_actions(self) -> list[Any]:
        obs = Observation(json.dumps(copy.deepcopy(self.base_obs)))
        return obs.legal_actions()

    def _get_mjx_obs(self, mjai_events):
        for mjai_event in mjai_events:
            mjai_event_type = mjai_event.get("type")
            if mjai_event_type == "start_kyoku":
                tehais = [to_mjx_tile(s) for s in mjai_event["tehais"][self.actor_id]]
                self.hai_offset = {to_mjx_tile(mjai_event["dora_marker"]): 1}
                for i, hai in enumerate(tehais):
                    tehais[i] += self.hai_offset.get(hai, 0)
                    self.hai_offset[hai] = self.hai_offset.get(hai, 0) + 1
                oya = mjai_event["oya"]
                rotated = [f"player_{(oya + i) % 4}" for i in range(4)]
                self.base_obs = {
                    "who": self.actor_id,
                    "publicObservation": {
                        "playerIds": rotated,
                        "initScore": {"tens": mjai_event["scores"], "round": (mjai_event["kyoku"] - 1) + (0 if mjai_event["bakaze"] == "E" else 4), "honba": mjai_event["honba"], "riichi": mjai_event["kyotaku"]},
                        "doraIndicators": [to_mjx_tile(mjai_event["dora_marker"])],
                        "events": [],
                    },
                    "privateObservation": {
                        "who": self.actor_id,
                        "initHand": {"closedTiles": tehais},
                        "drawHistory": [],
                        "currHand": {"closedTiles": sorted(tehais), "opens": []}
                    }
                }
            elif mjai_event_type == "tsumo":
                if self.actor_id == mjai_event["actor"]:
                    hai = to_mjx_tile(mjai_event["pai"])
                    hai_ = hai + self.hai_offset.get(hai, 0)
                    self.hai_offset[hai] = self.hai_offset.get(hai, 0) + 1
                    self.base_obs["privateObservation"]["drawHistory"].append(hai_)
                    self.base_obs["privateObservation"]["currHand"]["closedTiles"].append(hai_)
                    self.base_obs["privateObservation"]["currHand"]["closedTiles"].sort()
                row = {"type": "EVENT_TYPE_DRAW"}
                if mjai_event["actor"] > 0: row["who"] = mjai_event["actor"]
                self.base_obs["publicObservation"]["events"].append(row)
            elif mjai_event_type == "dahai":
                hai = to_mjx_tile(mjai_event["pai"])
                self.last_discard_actor = mjai_event["actor"]
                if self.actor_id == mjai_event["actor"]:
                    remove_candidates = [t for t in self.base_obs["privateObservation"]["currHand"]["closedTiles"] if to_mjx_tile(to_mjai_tile(t)) == hai]
                    if remove_candidates:
                        hai_ = remove_candidates[0]
                    else:
                        draw_h = self.base_obs["privateObservation"]["drawHistory"]
                        closed = self.base_obs["privateObservation"]["currHand"]["closedTiles"]
                        hai_ = (draw_h[-1] if draw_h and draw_h[-1] in closed else closed[-1])
                    self.base_obs["privateObservation"]["currHand"]["closedTiles"].remove(hai_)
                else:
                    hai_ = hai + self.hai_offset.get(hai, 0)
                    self.hai_offset[hai] = self.hai_offset.get(hai, 0) + 1
                row = {"type": ("EVENT_TYPE_TSUMOGIRI" if mjai_event.get("tsumogiri") else "EVENT_TYPE_DISCARD"), "tile": hai_}
                if mjai_event["actor"] > 0: row["who"] = mjai_event["actor"]
                self.base_obs["publicObservation"]["events"].append(row)
            elif mjai_event_type in ["chi", "pon", "ankan", "daiminkan"]:
                if mjai_event_type == "chi":
                    code, consumed = OpenCodeGen.from_mjai_chi(mjai_event, self.base_obs)
                    e_type = "EVENT_TYPE_CHI"
                elif mjai_event_type == "pon":
                    code, consumed = OpenCodeGen.from_mjai_pon(mjai_event, self.base_obs)
                    e_type = "EVENT_TYPE_PON"
                elif mjai_event_type == "ankan":
                    code, consumed = OpenCodeGen.from_mjai_kan(mjai_event, self.base_obs)
                    e_type = "EVENT_TYPE_CLOSED_KAN"
                else:
                    code, consumed = OpenCodeGen.from_mjai_kan(mjai_event, self.base_obs)
                    e_type = "EVENT_TYPE_OPEN_KAN"
                if self.actor_id == mjai_event["actor"]:
                    self.base_obs["privateObservation"]["currHand"]["opens"].append(code)
                    for t in consumed: self.base_obs["privateObservation"]["currHand"]["closedTiles"].remove(t)
                row = {"type": e_type, "open": code}
                if mjai_event["actor"] > 0: row["who"] = mjai_event["actor"]
                self.base_obs["publicObservation"]["events"].append(row)
            elif mjai_event_type == "reach":
                row = {"type": "EVENT_TYPE_RIICHI"}
                if mjai_event["actor"] > 0: row["who"] = mjai_event["actor"]
                self.base_obs["publicObservation"]["events"].append(row)
            elif mjai_event_type == "kakan":
                code, consumed, c_tile, p_code = OpenCodeGen.from_mjai_kakan(mjai_event, self.base_obs)
                if self.actor_id == mjai_event["actor"]:
                    self.base_obs["privateObservation"]["currHand"]["opens"].append(code)
                    self.base_obs["privateObservation"]["currHand"]["opens"].remove(p_code)
                    self.base_obs["privateObservation"]["currHand"]["closedTiles"].remove(c_tile)
                row = {"type": "EVENT_TYPE_ADDED_KAN", "open": code}
                if mjai_event["actor"] > 0: row["who"] = mjai_event["actor"]
                self.base_obs["publicObservation"]["events"].append(row)
            elif mjai_event_type == "dora":
                self.base_obs["publicObservation"]["events"].append({"type": "EVENT_TYPE_NEW_DORA", "tile": to_mjx_tile(mjai_event["dora_marker"])})

        obs_json = copy.deepcopy(self.base_obs)
        obs_json_str = Observation.add_legal_actions(json.dumps(obs_json, separators=(",", ":")))
        obs_json_filtered = json.loads(obs_json_str)
        curr_hand_set = set(self.base_obs["privateObservation"]["currHand"]["closedTiles"])
        filtered_actions = []
        for action in obs_json_filtered.get("legalActions", []):
            a_type = action.get("type", "")
            if (a_type == "" or a_type == "ACTION_TYPE_TSUMOGIRI") and "tile" in action:
                if action["tile"] in curr_hand_set: filtered_actions.append(action)
            else:
                filtered_actions.append(action)
        obs_json_filtered["legalActions"] = filtered_actions
        return Observation(json.dumps(obs_json_filtered))

    def _get_mjai_response(self, mjx_action):
        try:
            action_type = mjx_action.type()
        except ValueError:
            return json_dumps({"type": "none"})
        action_json = json.loads(mjx_action.to_json())
        if action_type in [ActionType.DISCARD, ActionType.TSUMOGIRI]:
            return json_dumps({"type": "dahai", "actor": self.actor_id, "pai": to_mjai_tile(action_json.get("tile", 0)), "tsumogiri": action_type == ActionType.TSUMOGIRI})
        if action_type == ActionType.RIICHI:
            return json_dumps({"type": "reach", "actor": self.actor_id})
        if action_type == ActionType.CLOSED_KAN:
            return json_dumps({"type": "ankan", "actor": self.actor_id, "target": self.actor_id, "consumed": [to_mjai_tile(tile.id()) for tile in mjx_action.open().tiles_from_hand()]})
        if action_type == ActionType.ADDED_KAN:
            op = mjx_action.open()
            return json_dumps({"type": "kakan", "actor": self.actor_id, "pai": to_mjai_tile(op.last_tile().id()), "consumed": [to_mjai_tile(tile.id()) for tile in op.tiles_from_hand()]})
        if action_type == ActionType.TSUMO:
            return json_dumps({"type": "hora", "actor": self.actor_id, "target": self.actor_id})
        if action_type == ActionType.ABORTIVE_DRAW_NINE_TERMINALS:
            return json_dumps({"type": "ryukyoku", "actor": self.actor_id})
        if action_type == ActionType.OPEN_KAN:
            op = mjx_action.open()
            return json_dumps({"type": "daiminkan", "actor": self.actor_id, "target": self.last_discard_actor, "pai": to_mjai_tile(op.stolen_tile().id()), "consumed": [to_mjai_tile(tile.id()) for tile in op.tiles_from_hand()]})
        if action_type == ActionType.RON:
            return json_dumps({"type": "hora", "pai": to_mjai_tile(action_json.get("tile", 0)), "actor": self.actor_id, "target": self.base_obs["publicObservation"]["events"][-1].get("who", 0)})
        if action_type == ActionType.CHI:
            op = mjx_action.open()
            return json_dumps({"type": "chi", "actor": self.actor_id, "target": self.actor_id - 1 if self.actor_id > 0 else 3, "pai": to_mjai_tile(op.stolen_tile().id()), "consumed": [to_mjai_tile(tile.id()) for tile in op.tiles_from_hand()]})
        if action_type == ActionType.PON:
            op = mjx_action.open()
            return json_dumps({"type": "pon", "actor": self.actor_id, "target": self.last_discard_actor, "pai": to_mjai_tile(op.stolen_tile().id()), "consumed": [to_mjai_tile(tile.id()) for tile in op.tiles_from_hand()]})
        return json_dumps({"type": "none"})

    def react(self, events_str: str) -> str:
        events = json.loads(events_str)
        if events[-1]["type"] in ["start_game", "end_kyoku", "end_game"]:
            return json_dumps({"type": "none"})
        obs = self._get_mjx_obs(events)
        if self.mjx_bot is None:
            import mortal_like_agent
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.mjx_bot = mortal_like_agent.MortalLikeAgent("mortal_like_agent_weight.pt", device=device)
        try:
            mjx_action = self.mjx_bot.act(obs)
        except Exception:
            return json_dumps({"type": "none"})
        return self._get_mjai_response(mjx_action)