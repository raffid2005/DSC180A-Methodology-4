# DSC 180A Methodology Assignment 4

**Name:** Raffi Dulgerian  
**Email:** [rdulgerian@ucsd.edu](mailto:rdulgerian@ucsd.edu)

**Section:** B?? – KV Cache Quantization for LLM Inference <!-- TODO: put your section code -->  
**Mentor:** Arya Mazumdar

---

**What is the most interesting topic covered in your domain this quarter?**

The most interesting topic was KIVI's observation that the keys and values in a transformer's KV cache have very different structures and should be quantized differently. Keys contain a few channels with outliers much larger than the rest, so quantizing them per-channel keeps those outliers from inflating the error of every other entry. Values have no such channels, and since each attention output is a weighted mix of value rows, quantizing them per-token keeps the error local to each token. It surprised me that a 2-bit cache could work at all, and that the answer to "how should we quantize this?" came from looking at the data rather than from a general rule. Reproducing it on Qwen3-1.7B, we also found that the value-side difference barely shows up in reconstruction error and only appears in the attention output. That taught me that which error metric you measure can change the conclusion.

**Describe a potential investigation you would like to pursue for your Quarter 2 Project.**

Our Quarter 1 Project measured quantization error from a single prefill pass and never generated any text. For Quarter 2, I would like to measure whether these errors actually matter downstream. I would build a KV cache quantizer into Qwen3's generation loop and compare 2-bit and 4-bit, per-channel and per-token settings on perplexity (e.g. WikiText-2) and on long-context tasks such as LongBench or needle-in-a-haystack retrieval. One question I am especially curious about is whether the "attention sink" heads that broke per-channel value quantization in Quarter 1 cause visible failures during generation, and whether keeping only the first token's keys and values in full precision fixes most of the problem cheaply. If that works, a related follow-up would be to measure the memory saved against the accuracy lost and find which layers can be compressed most aggressively.

**What is a potential change you'd make to the approach taken in your current Quarter 1 Project?**

The biggest change would be to use many prompts instead of one. All of our Quarter 1 results come from a single 759-token prompt, so we can't tell how much they would change with different text, lengths, or domains (code, math, conversation). I would run the same pipeline over a set of prompts of varying lengths and report confidence intervals for each error. I would also simulate decoding rather than only prefill: KIVI keeps the most recent tokens in full precision as a residual, and our analysis did not model how errors build up as new tokens are added to the cache over time.

**What other techniques would you be interested in using in your project?**

I would like to compare KIVI with other KV cache compression methods, such as KVQuant's non-uniform quantization with pre-RoPE key quantization, and rotation-based methods like QuaRot that use Hadamard transforms to spread outliers out before quantizing. I am also interested in token eviction methods (H2O, StreamingLLM), which drop unimportant tokens instead of compressing every token, and in whether eviction and quantization can be combined. On the systems side, I would like to write a fused dequantize-and-attend kernel in Triton, so we can measure real memory and speed improvements rather than only simulated error.
