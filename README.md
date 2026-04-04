# MortalStyleNet-mjx

> **Note:** This is a personal study project. Not intended for production use.

A riichi mahjong AI agent for [mjx](https://github.com/mjx-project/mjx), trained with supervised learning on Tenhou houou-level game logs.

## Overview

A simplified Mortal-inspired agent trained on 16,000 hanchan games from Tenhou houou-level players using supervised learning only.

## Files

| File | Description |
|------|-------------|
| `mortal_like_agent.py` | Model architecture and agent class |
| `mortal_style_feature.py` | Feature encoder (506-channel) |
| `requirements.txt` | Python dependencies |
| `smoke_test.py` | Usage example |

## Network

- **Architecture**: 1D Conv + Channel Attention + Mish + BatchNorm (MortalStyleNet)
- **Input**: 506-channel feature vector — see `mortal_style_feature.py`
- **Output**: 181-dimensional action logits

| Index | Action |
|-------|--------|
| 0–33 | Discard m1–rd |
| 34, 35, 36 | Discard m5(red), p5(red), s5(red) |
| 37–70 | Tsumogiri m1–rd |
| 71, 72, 73 | Tsumogiri m5(red), p5(red), s5(red) |
| 74–94 | Chi m1m2m3 – s7s8s9 |
| 95–103 | Chi with red 5 |
| 104–137 | Pon m1–rd |
| 138, 139, 140 | Pon m5(w/ red), s5(w/ red), p5(w/ red) |
| 141–174 | Kan m1–rd |
| 175 | Tsumo |
| 176 | Ron |
| 177 | Riichi |
| 178 | Kyuushu |
| 179 | No |
| 180 | Dummy |

Reference: [mjx action.h](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/action.h#L45-L61)

## Weights

Download from Google Drive:
https://drive.google.com/drive/folders/1nimmgp6KBEwywAVJsQTty-DQtdXL66Oy

## Usage

See `smoke_test.py`.

Requires mjx — Docker image available at:
https://github.com/gallantg762/mjx-docker

## Live

- mjai.app: https://mjai.app/users/gallantg762
- riichi.dev: https://riichi.dev/bots/74

## Strength

TBD