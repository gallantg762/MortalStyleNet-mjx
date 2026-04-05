# TCP client for connecting a Mahjong AI agent to an MJAI-compatible server.

import copy
import json
import socket
import sys
import time
from typing import Any, Optional
import mjai_gateway

BOT_NAME = "mortalstylenet"

_my_id = -1
_none_response = {"type": "none"}
_scores = [25000] * 4
_event_stack = []
_bot = mjai_gateway.MjxGateway(0, None)
_last_dahai_id = -1
_last_dahai_pai = ""
_last_tsumo_str = ""

def _reset_state() -> None:
    global _my_id, _scores, _event_stack, _bot, _last_dahai_id, _last_dahai_pai, _last_tsumo_str
    _my_id, _scores, _event_stack = -1, [25000] * 4, []
    _bot = mjai_gateway.MjxGateway(0, None)
    _last_dahai_id, _last_dahai_pai, _last_tsumo_str = -1, "", ""

def _json_dumps(json_data: dict) -> str:
    return json.dumps(json_data, separators=(",", ":"))

def parse(events_str: str) -> str:
    global _my_id, _scores, _event_stack, _bot, _last_dahai_id, _last_dahai_pai, _last_tsumo_str
    events = json.loads(events_str)
    e_type = events["type"]

    if e_type == "hello":
        return _json_dumps({"type": "join", "name": BOT_NAME, "room": "default"})
    elif e_type == "start_game":
        _my_id = events["id"]
        _bot.actor_id = _my_id
        return _json_dumps(_none_response)
    elif e_type in ["hora", "ryukyoku"]:
        if events.get("scores"): _scores = events["scores"]
        return _json_dumps(_none_response)
    elif e_type == "end_kyoku":
        return _json_dumps(_none_response)
    elif e_type == "end_game":
        _reset_state()
        return _json_dumps(_none_response)
    elif e_type == "start_kyoku":
        _event_stack.clear()
        e = copy.deepcopy(events)
        e["scores"] = _scores
        _event_stack.append(e)
        return _json_dumps(_none_response)
    elif e_type == "tsumo":
        _event_stack.append(events)
        _last_tsumo_str = events["pai"]
        return _think() if events["actor"] == _my_id else _json_dumps(_none_response)
    elif e_type == "reach":
        _event_stack.append(events)
        return _think() if events["actor"] == _my_id else _json_dumps(_none_response)
    elif e_type == "reach_accepted":
        _event_stack.append(events)
        if events.get("scores"): _scores = events["scores"]
        return _json_dumps(_none_response)
    elif e_type == "dahai":
        _event_stack.append(events)
        _last_dahai_id, _last_dahai_pai = events["actor"], events["pai"]
        return _think() if (events["actor"] != _my_id and events.get("possible_actions")) else _json_dumps(_none_response)
    elif e_type in ["ankan", "daiminkan", "kakan", "dora"]:
        _event_stack.append(events)
        return _json_dumps(_none_response)
    elif e_type in ["pon", "chi"]:
        _event_stack.append(events)
        return _think() if events["actor"] == _my_id else _json_dumps(_none_response)
    return _json_dumps(_none_response)

def _think() -> str:
    global _event_stack, _my_id, _bot, _last_dahai_id, _last_tsumo_str
    res = _bot.react(json.dumps(_event_stack, separators=(",", ":")))
    _event_stack.clear()
    res_ev = json.loads(res)
    if res_ev["type"] == "ryukyoku": res_ev["reason"] = "kyushukyuhai"
    if res_ev["type"] in ["pon", "daiminkan", "chi"]: res_ev["target"] = _last_dahai_id
    if res_ev["type"] == "hora":
        if res_ev.get("target") == _my_id: res_ev["pai"] = _last_tsumo_str
        else: res_ev["pai"] = _last_dahai_pai
    return json.dumps(res_ev, separators=(",", ":"))

def _recv_line(s: socket.socket, buf: bytearray) -> tuple[str, bytearray]:
    while b"\n" not in buf:
        chunk = s.recv(4096)
        if not chunk: raise ConnectionError("connection closed")
        buf += chunk
    idx = buf.index(b"\n")
    line, buf = buf[:idx].decode("utf-8"), buf[idx + 1:]
    return line, buf

def tcp() -> None:
    host, port = "127.0.0.1", 11600
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            buf = bytearray()
            while True:
                line, buf = _recv_line(s, buf)
                if not line: continue
                snd = parse(line)
                s.send((snd + "\n").encode("utf-8"))
        except (ConnectionError, socket.error):
            pass
        finally:
            s.close()
        time.sleep(0.5)

if __name__ == "__main__":
    tcp()