# MortalStyleNet-mjx

Riichi mahjong AI agent for [mjx](https://github.com/mjx-project/mjx).  
Architecture: MortalStyleNet (1D Conv + Channel Attention), trained on 16,000 Tenhou Houou-level games via supervised learning.

> **Note:** This is a personal study project. mjx has known compatibility issues with Tenhou; other simulators are recommended for training and evaluation.

## Performance

**Tenhou** — Advanced / East-South (上南), 2 Dan, avg. rank 2.45  
[[Log 1]](https://tenhou.net/3/?log=2026040609gm-0089-0000-443c33c4&tw=2) [[Log 2]](https://tenhou.net/3/?log=2026040610gm-0089-0000-05015719&tw=2) [[Log 3]](https://tenhou.net/3/?log=2026040610gm-0089-0000-d354a97a&tw=0)

**Discard Accuracy**

| Model | Accuracy |
| :--- | :--- |
| MortalStyleNet-mjx | 73.1% |
| Suphx | 76.7% |

**vs Bots** ([akochan](https://github.com/Apricot-S/akochan-docker), [mjai-manue](https://github.com/gimite/mjai-manue))

| | akochan | MortalStyleNet-mjx | mjai-manue (×2) |
| :--- | :--- | :--- | :--- |
| Tonpu (500g) | **2.196** | 2.316 | 2.752 |
| Tonnan (200g) | **2.151** | 2.343 | 2.753 |

## Architecture

- **Input**: 506-channel feature vector
- **Output**: 181-dimensional action logits ([mjx action.h](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/action.h#L45-L61))

## Usage

1. Download weights from [Google Drive](https://drive.google.com/drive/folders/1nimmgp6KBEwywAVJsQTty-DQtdXL66Oy)
2. Use [mjx-docker](https://github.com/gallantg762/mjx-docker) for a consistent environment

## Live

- [mjai.app](https://mjai.app/users/gallantg762)
- [riichi.dev](https://riichi.dev/bots/74)