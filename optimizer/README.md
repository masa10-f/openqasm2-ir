# QASM Optimizer using pytket

pytket を使用して OpenQASM 2.0 回路を最適化するツールです。

## インストール

```bash
# pytket を含む optimizer 依存関係のインストール
pip install -e ".[optimizer]"
```

## 使い方

### コマンドラインツールとして

インストール後、以下のコマンドが使用可能になります：

```bash
# 基本的な使い方
tket-optimize-qasm input.qasm output_optimized.qasm

# SWAP ゲートの挿入を無効化
tket-optimize-qasm input.qasm output_optimized.qasm --no-swaps

# 詳細な出力
tket-optimize-qasm input.qasm output_optimized.qasm -v
```

### スタンドアロンスクリプトとして

```bash
# リポジトリのルートから
python tket_opt_qasm.py input.qasm output_optimized.qasm

# SWAP ゲートの挿入を無効化
python tket_opt_qasm.py input.qasm output_optimized.qasm --no-swaps
```

### Python モジュールとして

```python
from optimizer import optimise_qasm_with_tket

# 基本的な使い方
optimise_qasm_with_tket("input.qasm", "output_optimized.qasm")

# オプションを指定
optimise_qasm_with_tket(
    "input.qasm",
    "output_optimized.qasm",
    allow_swaps=False  # SWAP ゲートの挿入を無効化
)
```

### モジュールとして実行

```bash
# Python モジュールとして実行
python -m optimizer input.qasm output_optimized.qasm
python -m optimizer input.qasm output_optimized.qasm --no-swaps
```

## 最適化パス

このオプティマイザーは以下の pytket 最適化パスを順番に適用します：

1. **FullPeepholeOptimise**: 包括的な最適化パス
   - Clifford 簡約
   - 1-3 量子ビットゲートシーケンスのパターンマッチング
   - 回転ゲートのマージ
   - 交換則に基づく最適化

2. **RemoveRedundancies**: 冗長なゲートの削除
   - ゲートと逆ゲートのペア
   - 角度ゼロの回転ゲート
   - その他の冗長な操作

## パラメータ

### `allow_swaps` (デフォルト: `True`)

- `True`: 最適化中に SWAP ゲートの挿入を許可します
- `False`: SWAP ゲートを挿入せず、量子ビットマッピングを保持します

MBQC や特定の量子ビットマッピングが必要な場合は `allow_swaps=False` を使用してください。

## 例

### QASMBench の一括最適化

```bash
# すべての QASM ファイルを最適化
for f in qasmbench/*.qasm; do
    tket-optimize-qasm "$f" "optimized/$(basename $f)"
done
```

### Python スクリプトでの一括処理

```python
from pathlib import Path
from optimizer import optimise_qasm_with_tket

input_dir = Path("qasmbench")
output_dir = Path("optimized")
output_dir.mkdir(exist_ok=True)

for qasm_file in input_dir.glob("*.qasm"):
    output_file = output_dir / qasm_file.name
    optimise_qasm_with_tket(qasm_file, output_file)
    print(f"Optimized: {qasm_file.name}")
```

## 技術詳細

### サポートされるゲートセット

pytket は以下を含む幅広いゲートをサポートしています：

- 単一量子ビットゲート: H, X, Y, Z, S, T, Rx, Ry, Rz など
- 2量子ビットゲート: CX (CNOT), CZ, SWAP, CRx, CRy, CRz など
- 多量子ビットゲート: CCX (Toffoli), CCZ など

### カスタム最適化

特定のゲートセットや最適化目標がある場合、`optimizer/tket_optimizer.py` の
`SequencePass` 構成をカスタマイズできます：

```python
from pytket.passes import (
    DecomposeBoxes,
    FullPeepholeOptimise,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
)

# カスタム最適化パス
custom_pass = SequencePass([
    DecomposeBoxes(),
    SynthesiseTket(),  # 特定のゲートセットに分解
    FullPeepholeOptimise(allow_swaps=False),
    RemoveRedundancies(),
])
```

## 参考リンク

- [pytket Documentation](https://docs.quantinuum.com/tket/)
- [pytket.qasm](https://docs.quantinuum.com/tket/api-docs/qasm.html)
- [Compilation - pytket user guide](https://docs.quantinuum.com/tket/user-guide/manual/manual_compiler.html)

## ライセンス

このプロジェクトのライセンスに従います。
