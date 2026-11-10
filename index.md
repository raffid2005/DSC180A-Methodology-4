# DSC 180A Methodology Assignment 4

**Name:** Raffi Dulgerian  
**Email:** [rdulgerian@ucsd.edu](mailto:rdulgerian@ucsd.edu)

**Section:** D39 – Quantization in ML Models
**Mentor:** Arya Mazumdar

---

**What is the most interesting topic covered in your domain this quarter?**

The most interesting topic was KIVI's observation that the keys and values in a transformer's KV cache have very different structures and should be quantized differently. Keys contain a few channels with outliers much larger than the rest, so quantizing them per-channel keeps those outliers from inflating the error of every other entry. Values have no such channels, and since each attention output is a weighted mix of value rows, quantizing them per-token keeps the error local to each token. It surprised me that a 2-bit cache could work at all, and that the answer to "how should we quantize this?" came from looking at the data rather than from a general rule. Reproducing it on Qwen3-1.7B, we also found that the value-side difference barely shows up in reconstruction error and only appears in the attention output. That taught me that which error metric you measure can change the conclusion.

**Describe a potential investigation you would like to pursue for your Quarter 2 Project.**

For Quarter 1, we measured quantization error from a single prefill pass without generating any text. For Quarter 2, I would like to test whether these errors actually affect generation quality. I would integrate a KV cache quantizer into Qwen3's generation loop and compare 2-bit and 4-bit, per-channel and per-token quantization using perplexity on WikiText-2 and long-context tasks such as LongBench or needle-in-a-haystack retrieval. I am especially curious whether the attention sink heads that caused problems with per-channel value quantization in Quarter 1 lead to noticeable failures during generation, and whether keeping the first token's keys and values in full precision fixes most of the issue with little overhead. If so, I would also like to measure the tradeoff between memory savings and accuracy loss and identify which layers can be compressed most aggressively.

**What is a potential change you'd make to the approach taken in your current Quarter 1 Project?**

The biggest change would be to test many prompts instead of just one. All our Quarter 1 results come from a single 759-token prompt, so we cannot tell how much the results change across different text, lengths, or domains such as code, math, and conversation. I would run the same pipeline on prompts of varying lengths and report confidence intervals for each error metric. I would also simulate decoding instead of only prefill. Since KIVI keeps the most recent tokens in full precision as a residual, our analysis does not capture how quantization errors accumulate as new tokens are added to the cache over time.

**What other techniques would you be interested in using in your project?**

I would like to compare KIVI with other KV cache compression methods, such as KVQuant, which uses non-uniform quantization and pre-RoPE key quantization, and rotation-based methods like QuaRot, which use Hadamard transforms to spread outliers before quantization. I am also interested in token eviction methods like H2O and StreamingLLM, which remove less important tokens instead of compressing every token, and whether eviction and quantization can be combined. On the systems side, I would like to implement a fused dequantization-and-attention kernel in Triton to measure actual memory and speed improvements rather than just simulated quantization error.
