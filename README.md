# MortalStyleNet-mjx

Riichi mahjong AI agent for [mjx](https://github.com/mjx-project/mjx).  

Architecture: MortalStyleNet (1D Conv + Channel Attention), trained on 16,000 Tenhou Houou-level games via supervised learning.

This is just a personal study project.

## Strength

### Tenhou

TBD

### vs Bots ([akochan](https://github.com/Apricot-S/akochan-docker), [mjai-manue](https://github.com/gimite/mjai-manue))

| | akochan | MortalStyleNet-mjx | mjai-manue (×2) |
| :--- | :--- | :--- | :--- |
| Tonnan (220game) | **2.144** | 2.258 | 2.799 |

### Discard Accuracy

| Model | Accuracy |
| :--- | :--- |
| MortalStyleNet-mjx | 73.1% |
| Suphx | 76.7% |

## Architecture

- **Input**: 506-channel feature vector
- **Output**: 181-dimensional action logits ([mjx action.h](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/action.h#L45-L61))

## Usage

1. Download weights from [Google Drive](https://drive.google.com/drive/folders/1nimmgp6KBEwywAVJsQTty-DQtdXL66Oy)
2. Use [mjx-docker](https://github.com/gallantg762/mjx-docker) for a consistent environment

## Live

- [mjai.app](https://mjai.app/users/gallantg762)
- [riichi.dev](https://riichi.dev/bots/74)