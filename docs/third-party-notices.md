# Third-party notices

## DeepSeek-Prover-V2-7B tokenizer

The tokenizer archived under [`configs/`](../configs/) was initialized from DeepSeek-Prover-V2-7B and saved after adding the `<proof>` and `<eofill>` special tokens. The archive compresses the tokenizer JSON and includes the matching tokenizer configuration and chat template. The upstream Hugging Face distribution includes a DeepSeek copyright notice and an MIT license.

[Upstream tokenizer distribution](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B/tree/main) · [DeepSeek MIT license and copyright notice](licenses/deepseek-prover-v2-7b-mit.txt) · [DeepSeek model-use agreement](https://github.com/deepseek-ai/DeepSeek-Prover-V2/blob/main/LICENSE-MODEL)

## FormalMATH

The FormalMATH rows in [`data/`](../data/) and [`results/`](../results/) contain theorem statements from **FormalMATH-All**, published by CUHK SphereLab and described by Zhouliang Yu and coauthors. SketchFill uses those statements to generate and screen complete proofs, extract sketches and fragments, and record completion attempts. The upstream dataset card identifies the dataset's license as MIT.

[Upstream dataset and license declaration](https://huggingface.co/datasets/SphereLab/FormalMATH-All) · [Dataset attribution and MIT license text](licenses/formalmath-all-mit.txt) · [FormalMATH paper](https://arxiv.org/abs/2505.02735)

## MiniF2F

The MiniF2F rows in [`data/`](../data/) and [`results/`](../results/) contain Lean 4 versions of MiniF2F theorem statements. SketchFill uses them to generate proof candidates, extract sketches and fragments, and evaluate assembled completions. The original MiniF2F Lean material is distributed under Apache-2.0. The MiniF2F Lean 4 port referenced in the research setup includes an MIT license.

The original Lean source files carry this notice:

> Copyright (c) 2021 OpenAI. All rights reserved.
>
> Authors: Kunhao Zheng, Kudzo Ahegbebu, Stanislas Polu, David Renshaw, OpenAI GPT-f

[Original MiniF2F Lean material](https://github.com/openai/miniF2F/tree/main/lean) · [Upstream Apache-2.0 license](licenses/minif2f-lean-apache-2.0.txt) · [Referenced Lean 4 port](https://github.com/yangky11/miniF2F-lean4) · [Port MIT license and copyright notice](licenses/minif2f-lean4-port-mit.txt)
