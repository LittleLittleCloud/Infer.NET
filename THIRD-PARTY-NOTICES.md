# Third-party notices

## llama2.c and tokenizer compatibility

The retired legacy FP32 checkpoint reader, tokenizer behavior and transformer implementation were developed with reference to Andrej Karpathy's llama2.c (revision 350e04fe35433e6d2941dce5a1f53308f87058eb). The historical attribution is retained for any shared reference-derived portions. The Metal bridge/kernel is an original implementation.

MIT License

Copyright (c) 2023 Andrej

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## RazorConsole

The optional UI uses RazorConsole.Core 0.6.0 (MIT). Its package and transitive dependency notices remain with their upstream packages. No dotLLM source is incorporated.

## Qwen3.6 checkpoint and independent MLX reference

The separately downloaded `mlx-community/Qwen3.6-35B-A3B-5bit` quantization at
revision `c90c1a8b7e7a6a9e39e549f0361a96eb05bbb499` derives from Qwen's
Apache-2.0 model. Its upstream license and quantized-file provenance are retained
in `models/qwen3.6-35b-a3b-5bit-mlx/`. The same model’s 8bit checkpoint and its Apache-2.0 license/provenance are retained in `models/qwen3.6-35b-a3b-8bit-mlx/`. Binaries are not committed. MLX/MLX-LM
and their Python dependencies are installed only into an ignored external
benchmark environment; their upstream licenses apply. No MLX implementation
source is incorporated into the .NET engine. The original .NET implementation
reads the same affine checkpoint directly. MLX-LM 0.32.2 was consulted to
interpret the checkpoint's sanitized normalization/conv layout and as an
independent numerical oracle; it is not a product runtime dependency. See
`docs/QWEN36-OWN-BENCHMARK.md` and `docs/MLX-BASELINE.md`.

The reference implementation's upstream notice is retained for attribution:

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
