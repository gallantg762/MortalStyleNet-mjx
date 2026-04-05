# MortalStyleNet-mjx

A riichi mahjong AI agent for [mjx](https://github.com/mjx-project/mjx).

> **Note:** This is a personal study project and is not intended for production use.

## Overview
- **Core Architecture**: MortalStyleNet (1D Conv + Channel Attention)
- **Current Method**: Supervised Learning using 16,000 Tenhou Houou-level games.

## Strength

### Discard Accuracy
| Model | Accuracy |
| :--- | :--- |
| **MortalStyleNet-mjx** | **73.1%** |
| Suphx | 76.7% |

### vs Other Bots
Matches against [akochan](https://github.com/Apricot-S/akochan-docker) and [mjai-manue](https://github.com/gimite/mjai-manue).

#### Tonpu (500 games)
| Bot | Average Rank |
| :--- | :--- |
| 👑 akochan | 2.196 |
| **MortalStyleNet-mjx** | **2.316** |
| mjai-manue (x2) | 2.752 |

#### Tonnan (200 games)
| Bot | Average Rank |
| :--- | :--- |
| 👑 akochan | 2.151 |
| **MortalStyleNet-mjx** | **2.343** |
| mjai-manue (x2) | 2.753 |

## Architecture
- **Input**: 506-channel feature vector.
- **Output**: 181-dimensional action logits.
    - Follows the [mjx action.h](https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/include/mjx/internal/action.h#L45-L61) definition.
- **Inference**: GPU recommended for high-speed decision making.

## Files
- `mortal_like_agent.py`: Model architecture and agent class.
- `mortal_style_feature.py`: Feature encoder.
- `mjai_tcp_client.py`: TCP client for connecting to Mjai servers (e.g., mjai.app).
- `mjai_gateway.py`: Protocol converter (Mjai ↔ mjx).
- `test.py`: Usage examples.

## Getting Started
1. **Weights**: Download pre-trained SL weights from [Google Drive](https://drive.google.com/drive/folders/1nimmgp6KBEwywAVJsQTty-DQtdXL66Oy).
2. **Environment**: It is recommended to use the [mjx-docker](https://github.com/gallantg762/mjx-docker) image for a consistent setup.

## Live Demo
- [mjai.app](https://mjai.app/users/gallantg762)
- [riichi.dev](https://riichi.dev/bots/74)