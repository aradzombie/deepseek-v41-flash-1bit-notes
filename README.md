# DeepSeek V4.1 Flash at 1-bit on a 64 GB gaming PC

I run the 233 GB 1-bit DeepSeek V4.1 Flash 1017 (Unsloth UD-IQ1_S, the files from before their Oct 28 revision) in llama.cpp under WSL2, on a gaming PC with 64 GB of RAM and a 16 GB card.

- Side by side in one test run, with the RAM at 3600: the plain build did 4.37 tokens a second, and four lossless switches did 8.80 (2.02x). The words were the same. The four switches are expert fetch-ahead, AVX-512 kernels, in-RAM experts first and no pre-load at start (tricks 1 to 4 below). "Plain build" means my own patched binary with those switches off and the same flags, not stock llama.cpp. I didn't time an unpatched build in that run.
- Faster RAM, a faster drive and more tuning brought it to about 11.5 tokens a second, still with the same words. Measured: 11.36 for the newest set-up, with about one Windows core busy in the background, and 11.47 to 11.60 for the set-up just before it.
- Start-up went from 87 to 96 s down to 12 s.
- On a second, smaller PC (32 GB of RAM, 8 GB card) a subset of the same switches took it from 0.11 to 0.87 tokens a second, same words. That was two plain runs and one run with the switches, not paired blocks.

Everything here was measured on my PCs unless it says untested or "my reading". The patches are not public yet. They go up later, after a review. I'm also building a setup kit so other people can do this on their own PC. The kit's build is still in progress and it is not released.

Who did what: Claude agents (Anthropic) wrote the patches, built llama.cpp, ran every test and kept the notes. I directed the work, chose what to try and checked the results. Other Claude models reviewed the patches. No llama.cpp developer has looked at them yet.

Most tricks here already exist in another project or paper, credited by name below. What this write-up adds is the numbers: each trick measured on this model and this PC in its own A/B, with the output checked word for word.

## Short version

Each line is its own A/B inside one test run, unless it says otherwise. The words were identical in every one.

| Step | RAM | Drive | tok/s, before and after | Gain | Interval |
|---|---|---|---|---|---|
| Plain build to four lossless switches | 3600 | Gen4 | 4.37 to 8.80 | 2.02x | 90%: +99.4% to +105.3% |
| Faster RAM, same sticks (BIOS update and a BIOS setting) | 3600 to 6000 | Gen4 | 8.74 to 9.59 | +9.7% | none: one block against the night before, so treat it as rough |
| 2 fetch-ahead helpers instead of 4, and `--poll 0` | 6000 | Gen4 | 9.43 to 9.82 | +4.2% | 90%: +2.9% to +5.4% |
| Queued card hand-overs (my patch) | 6000 | Gen4 | 9.71 to 10.08 | +3.9% | 90%: +2.8% to +4.9% |
| Model moved from a Gen4 to a Gen5 NVMe drive | 6000 | Gen4 to Gen5 | 9.90 to 11.53 | +16.5% | 90%: +15.6% to +17.5% |
| 9 math threads instead of 8 | 6000 | Gen5 | 11.47 to 11.60 | +1.1% | 90%: +0.2% to +2.0% |
| IQ1_S multi-row kernel (my patch), about one busy Windows core all run | 6000 | Gen5 | 11.03 to 11.36 | +3.0% | 95%: +1.7% to +4.3% |

- The before and after numbers come from different days, so they don't chain exactly. Background load moves the single numbers. The gains are paired inside each run, so they hold up better.
- The newest set-up has all of the above. It read 11.36 with about one Windows core busy in the background. I haven't timed it on a quiet PC.
- I didn't re-time the plain build at RAM 6000 or on the Gen5 drive. So I give no "vs plain" number for the newer rows.
- Start-up: the server loads in 12 s instead of 87 to 96 s, with the same writing speed (trick 4 below).
- There is also a lossy option (fewer experts in light layers). It changes the words, so it's off by default. Numbers are further down.

## Set-up

### Hardware

Main PC:

| Part | What I have |
|---|---|
| CPU | 8-core, 16-thread desktop Ryzen (Zen 5, one CCD) |
| RAM | 64 GB DDR5, 4 x 16 GB. At 3600 up to Oct 30, at 6000 since then (after a BIOS update). They passed MemTest86 at 6000 (one pass, no errors). One pass is a short test, so I mostly rely on every answer at 6000 matching the 3600 answers |
| GPU | NVIDIA card with 16 GB |
| Drive | Gen4 NVMe up to Nov 9, Gen5 NVMe since then. The model sits on WSL's virtual disk |
| OS | Windows 11, Ubuntu in WSL2 |

Second PC (the setup kit test):

| Part | What it has |
|---|---|
| CPU | 8-core desktop Intel, no AVX-512 |
| RAM | 32 GB, of which Linux in WSL sees 16.7 GB |
| GPU | NVIDIA card with 8 GB |
| OS | Windows 11, Ubuntu 24.04 in WSL2 |

### Software

- llama.cpp b11870 (commit 74bca39, where PR 31086 added DeepSeek V4.1 support to main on Oct 25), with my patches on top. My first tests (Oct 23-24) ran on the draft PR 30817 instead. It closed unmerged on Oct 27 and PR 31086 replaced it. All the numbers here are from b11870. My patches are not moved to newer main yet.
- CUDA build for the card, run inside WSL2.
- WSL settings on the main PC: memory capped at 44 GB, autoMemoryReclaim off (so WSL keeps its page cache), and vm.swappiness 1 inside Ubuntu.

### The model

- unsloth/DeepSeek-V4.1-Flash-1017-GGUF, the UD-IQ1_S quant, 5 files, 233 GB. "1-bit" is Unsloth's name for the quant. IQ1_S is 1.5625 bits per weight, and other tensors in the file use more (IQ2_XXS is 2.0625).
- These are the files from before Unsloth's Oct 28 revision. The revision changed shards 1 to 4. One way to tell: shard 1 is 5,780,923 bytes in the set I used and 5,780,868 in the revised set.
- 40 layers, 384 experts per layer, 6 experts used per token.

### Server settings

- `-b 4096 -ub 4096 -c 8192 -np 1`, KV cache f16, mmap on. `-t 8` up to Nov 9, `-t 9` since Nov 10.
- Since Oct 31: `--poll 0` (no busy waiting) and 2 fetch-ahead helper threads. Before that, polling was at its default and there were 4 helpers.
- Fixed placement: `-fit off -ngl all`, plus `-ot` to send the experts of layers 2 to 39 (and layer 1's down experts) to the CPU side. The card holds everything else, about 15.2 GB of the 16.
- The experts of layers 2 to 39 live in the page cache or on the drive. With the stack on, about 155 MB a token is read from the drive.

## The tricks

### 1. Expert fetch-ahead right after routing

What it does: as soon as the router picks a layer's experts, helper threads send `MADV_WILLNEED` hints for exactly those expert slices, in 256 KiB pieces, with at most 512 MiB in flight. The math threads then find most pages already in RAM. With the stack on, about 92% of the bytes it asks for are already cached.

Why 256 KiB pieces: one WILLNEED call reads at most the larger of read_ahead_kb and max_sectors_kb, so one big hint gets cut short. On this PC that cap is 1280 KB, so each 256 KiB piece is read whole.

What I measured: on Oct 27, prose went from 3.44 to 6.91 tok/s (x2.02 paired) and copy jobs x1.71. On Oct 28 a mixed set went from 4.18 to 7.50 (+79%). Both were on an earlier version of my test harness, before the A/A noise check, so don't multiply them with the other rows. Since Oct 31: 2 helpers instead of 4, plus `--poll 0`, gave +4.2% (table above).

Public versions:
- ik_llama.cpp `--prefetch-experts`, by dmaivel ([PR 2101](https://github.com/ikawrakow/ik_llama.cpp/pull/2101), and the fix [PR 2667](https://github.com/ikawrakow/ik_llama.cpp/pull/2667)).
- LLMario [PR 19](https://github.com/mannysinghx/LLMario/pull/19): WILLNEED right after routing plus a mincore check.
- [Expert Sniper](https://huggingface.co/waltgrace/llama-cpp-expert-sniper) (waltgrace): one WILLNEED per chosen expert in a llama.cpp callback.
- Not everyone gains from read hints. [flash-moe](https://github.com/danveloper/flash-moe) lists its `F_RDADVISE` prefetch among its discarded tries, net 0%, on unified memory.

### 2. AVX-512 kernels for IQ1_S and IQ2_XXS

What it does: AVX-512 dot products for IQ1_S and IQ2_XXS, two of the low-bit quant types in this file. Main llama.cpp only has AVX2, AVX and plain C paths for these two (checked Oct 30). The results are byte-equal to the AVX2 code in my test harness: 0 of 218,790 cases differ for IQ1_S and 0 of 212,960 for IQ2_XXS.

What I measured: +10.7% (90% +10.0 to +11.4), 7.70 to 8.54 tok/s, at RAM 3600 on the Gen4 drive, with the fetch-ahead on in both arms.

Public versions:
- ik_llama.cpp [PR 2705](https://github.com/ikawrakow/ik_llama.cpp/pull/2705) and [PR 2676](https://github.com/ikawrakow/ik_llama.cpp/pull/2676) by fijam, and [PR 212](https://github.com/ikawrakow/ik_llama.cpp/pull/212) (IQ1_S kernels for Zen 4, AVX2 and NEON).
- Strata 0.2.4 has AVX-512 paths for Zen 4 and Zen 5.
- llama.cpp [PR 31420](https://github.com/ggml-org/llama.cpp/pull/31420) by R3nzTheCodeGOD (open on Nov 4) unrolls the AVX2 loops of these types.

So mine is a llama.cpp version of the same idea, measured on Zen 5. The IQ1_S kernel borrows ideas from ik_llama.cpp (MIT notice kept). The 512-bit IQ2_XXS kernel is new code, as far as I found.

### 3. In-RAM experts first

What it does: inside each expert step, the CPU does the experts whose pages are already in RAM first, and the ones still loading from the drive last. It knows which is which from the fetch-ahead's mincore check.

What I measured: +3.0% (90% +2.4 to +3.7), 7.64 to 7.87 tok/s, at RAM 3600 on the Gen4 drive, with the fetch-ahead on in both arms.

Public versions: [Strata 0.2.5](https://github.com/Niko1221/Strata/releases) (the CPU starts on the experts it already has while the rest are read from the SSD). Related: [HybriMoE](https://arxiv.org/abs/2504.05897), [llama.cpp RFC 24528](https://github.com/ggml-org/llama.cpp/discussions/24528) and [MoEpic](https://arxiv.org/abs/2509.08342).

### 4. No pre-load at start

What it does: skips MAP_POPULATE and the whole-file WILLNEED when the model loads.

What I measured: the same writing speed (-0.3%, 90% -1.5 to +1.0). The server loads in 12 s instead of 87 to 96 s, and the warm-up reads 55 GB instead of 174 GB. This one is for start-up time only.

Public versions: llama.cpp [PR 28055](https://github.com/ggml-org/llama.cpp/pull/28055) by Nekotekina (open on Nov 4) drops MAP_POPULATE on Linux. I did not find one for the whole-file WILLNEED half.

### 5. Queued card hand-overs

What it does: when data moves between the CPU side and the card, the copy is queued on the card's own stream instead of a blocking copy with a full wait in front of it. The next CPU step only waits where it has to. Same bytes, only the waits change.

What I measured: +3.9% (90% +2.8 to +4.9), 9.71 to 10.08 tok/s, 8 of 8 blocks, Gen4 drive, RAM 6000.

Public versions: I haven't looked for one.

### 6. 9 math threads instead of 8

What it does: one more math thread than cores. Nothing else changes.

What I measured: +1.1% (90% +0.2 to +2.0), 11.47 to 11.60 tok/s, 6 blocks, on the Gen5 drive. 10 threads was 15.9% slower in an earlier one-block test (Oct 31, RAM 6000, Gen4 drive), and 16 threads was 32% slower (RAM 3600).

### 7. IQ1_S multi-row kernel

What it does: computes the IQ1_S dot products of several weight rows against the same input in one call, so part of the input work is done once instead of once per row. The result for each row is bit for bit what the one-row AVX-512 kernel gives.

What I measured: +3.0% (95% +1.7 to +4.3), 11.03 to 11.36 tok/s, 5 blocks, every prompt faster, on the Gen5 drive with about one busy Windows core.

Public versions: it builds on the IQ1_S kernel in trick 2. I haven't checked whether a public multi-row IQ1_S kernel exists.

### 8. The faster drive and the faster RAM

These are hardware, not code. Same stack on both sides:
- Gen4 to Gen5 NVMe: +16.5% (90% +15.6 to +17.5), 9.90 to 11.53 tok/s, 8 of 8 blocks. The drive was 49% busy on the Gen4 and 34% on the Gen5, for about the same bytes a token.
- RAM 3600 to 6000, same 4 sticks: +9.7%, one block against the night before. The BIOS update also changed other things (it put the CPU's built-in graphics back on), so more than the RAM speed changed. RAM reads inside WSL went from 45.1 to 58.9 GB/s on a simple read test.

## The lossy option: fewer experts (off by default)

What it does: 4 or 5 experts instead of 6 in the 18 light layers (15, 19, 21, 23 to 34, and 37 to 39), keeping the router's own weights for the experts that stay.

What I measured, all on the lossless stack at RAM 3600 unless it says 6000. KL is the mean KL divergence against the 6-expert output on one fixed text (lower means closer to the 6-expert model). "Items right" is out of 44 small tasks a program can check (code, JSON, math, facts, copy jobs). The blind read is a separate Claude model grading prose answers with the names hidden.

| Arm | Speed vs 6 experts | KL | Items right (of 44) |
|---|---|---|---|
| 5 experts in light layers | +6.2% (95%: +1.2 to +11.4) | 0.0107 to 0.0138 | 43 |
| 4 experts in light layers | +15.6% and +17.9% in two runs (10.04 and 10.29 tok/s). At RAM 6000: 11.03 and 11.34, one block each, not a pair | 0.0248 | 42 (same as 6 experts) |

- The +6.2% was timed with renormalised weights (KL 0.0138). Keeping the router's weights picks the same experts and gave KL 0.0107, but I haven't timed that version on its own.
- These change the words, so they stay off unless a job kind passes the quality gate and a blind read. The 4-expert cut read worse on prose in a blind read. Code and JSON only passed small automatic checks (8 and 7 items), with no blind read.
- Untested: the lossy option on the newest set-up (queued hand-overs, Gen5 drive, 9 threads, multi-row kernel).

Public versions: Strata's "Faster" mode (5 of 6 experts in every layer, [docs](https://github.com/Niko1221/Strata/blob/main/docs/DEEPSEEK41.md)), [lifeidle](https://github.com/lifeidle/deepseek-v4.1-flash-5090-laptop-24gb-strata), ik_llama.cpp [PR 239](https://github.com/ikawrakow/ik_llama.cpp/pull/239) (Smart Expert Reduction), [LExI](https://arxiv.org/abs/2509.02753), [Alloc-MoE](https://arxiv.org/abs/2604.08133), and on not renormalising, [ZEDA](https://arxiv.org/html/2605.18643v2) and the [training-free halving paper](https://arxiv.org/abs/2609.04575).

## The second PC

A by-hand test of my setup kit on a smaller, older PC (hardware above), on Nov 9.
- Plain: 0.11 tok/s (two runs, 0.11 and 0.11). With the kit's settings: 0.87 tok/s (one run). About 8x.
- The answers were word for word the same on all 3 prompts (128 tokens each).
- Load time: 82.4 s and 94.3 s plain, 18.0 s with the kit's settings.
- The kit's settings there are a subset of the main PC's: fetch-ahead, in-RAM experts first, no pre-load at start, queued card hand-overs, no busy polling. That CPU has no AVX-512, so tricks 2 and 7 don't apply.
- These were single runs, not paired blocks like on the main PC, so there is no interval.

## The setup kit (in progress)

The kit is a guide, a build script and a self-test, so someone can run this on their own PC. Its build is still in progress and it is not released. On the second PC the guide worked through the model download. Two things still need fixing before anyone else uses it: the build stops on a PC that also has NVIDIA's CUDA toolkit for Windows installed, and the self-test took 2 hours 18 minutes against the 30 minutes it promised.

## Where the time goes now

Measured on the main PC on Nov 10, before the multi-row kernel: about 90 ms a token. The CPU part is about 64 ms (expert math about 48, drive waits about 10.5). The card does about 19.3 ms of work, and the CPU waits for it. Hand-overs are about 4.5 ms. The CPU and the card don't run at the same time.

## How it was tested

- **Paired A/B in blocks.** Each block starts a fresh server for every arm, in ABBA order for two arms (a balanced order for more). Slow drift during the night cancels out inside a block.
- **Prompts.** 8 prompts x 256 tokens, prose and copy jobs (jobs that mostly repeat their input, like fixing spelling). Temperature 0, fixed length. The page cache is not dropped between servers, because that's how the PC runs day to day.
- **Clocks.** Tok/s is the server's own timing. Gains come from the gaps between streamed tokens, paired per prompt inside each block, so dividing two printed averages can differ in the last digit.
- **A/A check.** The same build against an identical copy of itself read +0.0% (90% -0.7% to +0.7%). That check also caught Linux swapping out the server's own memory at the default swappiness of 60. Setting it to 1 fixed it.
- **Servers get thrown out and re-run** when the PC was busy: swap-ins, other Windows CPU over 2 cores, the card's 3D engine waking up, or a Windows disk burst.
- **Identical-output gate.** A trick only counts as lossless if the text is identical at every kept server.
- **Guesses written down before each run,** then checked against the result.

Words used here:
- **Block:** one round where every arm gets a fresh server, in a balanced order. **Arm:** one set-up in an A/B.
- **A/A:** the same build against itself, to see the noise.
- **Copy jobs:** prompts that mostly repeat their input, like fixing spelling.
- **Light layers:** the 18 layers where the 6th expert carries the least routing weight.
- **Math threads:** llama.cpp's compute threads (`-t`). **Helpers:** my fetch-ahead threads.
- **Hand-overs:** the copies between the CPU side and the card, each token.
- **WILLNEED, mincore, MAP_POPULATE:** Linux calls to ask for pages ahead, check which pages are in RAM, and pre-load a whole file.

## What did not work

All on the main PC. Most of these were short tests: 1 to 2 blocks, or single runs on unpatched llama.cpp or with only the fetch-ahead, before the full stack. A few ran on the early draft build (Oct 24): dense placement, batch 8192, the WSL cap and the second PC over RPC. Read them as rough.

| Idea | Result | Note |
|---|---|---|
| 16 threads instead of 8 (SMT) | -32% | with the fetch-ahead on |
| 10 math threads instead of 8 | -15.9% | one block |
| Fetch-ahead level 2 (guess the next layer's experts) | -3.7% on the Gen4 (95%: -5.4 to -2.0); -2.0% on the Gen5 (95%: -4.7 to +0.9) | on the Gen4 it read 220 MB a token against 155. The Gen5 run stopped in block 5 with 4 valid blocks, so not settled |
| 4 fetch-ahead helpers instead of 2, on the Gen5 | -1.9% (95%: -4.4 to +0.8) | 5 blocks, stopped early, not settled |
| 1024 KiB fetch pieces instead of 256, on the Gen5 | -2.3% (95%: -4.4 to -0.1) | slower |
| 1 MB and 128 KB fetch requests, 1 helper, helpers at lower priority, pinning threads | between -4.0% and +0.5% | short 2-block screens, none settled |
| CUDA graph optimisation (`GGML_CUDA_GRAPH_OPT=1`) | -0.1% | no effect |
| Experts held on the card (my own patch) | -2.4% | and the words changed, so out |
| llama.cpp's card expert cache (PR 29887), 4 GB | about 0.6x | my reading: the card room it takes costs more than it gives back here |
| A second PC over 1 Gbit (llama.cpp RPC) | -23% to -62% | |
| n-gram drafts | -15% to no gain | drafts accepted only 24-45% of the time |
| DeepSeek's draft model (8 GB) | did not fit | my memory guard stopped all four tries (free RAM fell to 0.8-4 GB) |
| Experts off the card (dense placement) | -10% | |
| Batch 8192 | reading faster, writing -22% | 32 GB set-up |
| WSL cap of 48 GB out of 64 | nearly locked up the PC | 44 GB works |

## Untested

- Long prompt reading with the stack on.
- The newest set-up on a quiet PC.
- The plain build at RAM 6000 or on the Gen5 drive.
- The lossy option on the newest set-up.
- Native Linux instead of WSL2.
- Unsloth's revised files (after Oct 28).
- Newer llama.cpp main.
- ik_llama.cpp on this model. Its source lists the `deepseek41` architecture (checked Nov 4), but I haven't tried it.
- The second PC with paired blocks.

## Limits

- One main PC, warm cache, short context (8192), temperature 0, 8 prompts of 256 tokens each with end-of-text ignored.
- The RAM step has no same-run partner, and the BIOS update changed other things too.
- The 1-bit file has its own quality cost against the full model. Unsloth's [guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash) lists KL 0.457 on wiki1100 and 0.364 on long17k for UD-IQ1_S (the page does not say which file version). All my KL numbers are against the 1-bit file with 6 experts, so they sit on top of that.
- The patches were written by Claude agents and reviewed by other Claude models. No outside developer has reviewed them.

## How to try it (outline)

The patches are not public yet. Without them you can still test unpatched b11870 with these settings. I only timed `-t 9` and `--poll 0` with my patches on, so on unpatched b11870 they are untested.

1. Windows 11 with WSL2 and Ubuntu. On a 64 GB PC, set WSL's memory to about 44 GB and turn autoMemoryReclaim off in .wslconfig. Inside Ubuntu, set `vm.swappiness=1` in a sysctl.d file.
2. Install NVIDIA's CUDA toolkit for WSL (WSL uses the Windows driver).
3. Check out llama.cpp at commit 74bca39 (release b11870) and build with CUDA (`-DGGML_CUDA=ON`, with `CMAKE_CUDA_ARCHITECTURES` set for your card).
4. Download the 5 UD-IQ1_S files (233 GB) to an NVMe, inside WSL's disk. To get the files I used, download with `--revision 60ad3651641b5c97679c807e064ab1d7a3ca1455` (in that revision shard 1 is 5,780,923 bytes, the same as my files). A plain download gives the revised files.
5. Close games and launchers. Check the card is close to empty.
6. Start `llama-server` on shard 1 of 5 with `-t 9 -b 4096 -ub 4096 -c 8192 -np 1 -fit off -ngl all --poll 0` and this `-ot` rule: `-ot "blk\.(1)\.ffn_(down)_(ch|)exps=CPU,blk\.(2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39)\.ffn_(down|gate|up)_(ch|)exps=CPU"`.
7. Test: run the same prompts at temperature 0 with your change off and on, alternating servers, and compare the texts word for word.

## What's next

The card part (about 19 ms a token) is the biggest block left. Then the patches go public after a review, and the setup kit once its build is done.

## Others, on different PCs

Their numbers, as they reported them. Not like for like: engines, files, expert counts and PCs differ. Listed by name, not ranked by speed.

| Who | PC | File | tok/s |
|---|---|---|---|
| [helgard-orlm](https://github.com/helgard-orlm/deepseek-v41-flash-8gb) | 8 GB card, 31 GiB RAM | official FP8 and FP4 weights, about 510 GB | 1.6 (drive only) to 2.4 (RAM cache) |
| [JigSawPT](https://github.com/ggml-org/llama.cpp/discussions/28766) | RTX 5090, 126 GiB RAM | MXFP4, 502 GB | 5.1 (21 on content already in its cache) |
| [lifeidle](https://github.com/lifeidle/deepseek-v4.1-flash-5090-laptop-24gb-strata) (on Strata) | 24 GB laptop card, 64 GB RAM | DASLab 2.00-bit | 11.5 (6 experts), 13.6 (5) |
| [Project Maya](https://dev.to/seppegadeyne/run-deepseek-v41-flash-748b-locally-with-project-maya-on-one-rtx-5090-4i4a) | RTX 5090, 128 GB RAM | its own 2-3 bit, about 340 GB | 24-26.7, no drafts (versions 1.1.16 to 1.1.27) |
| [Strata](https://github.com/Niko1221/Strata) | RTX 5070 12 GB, 64 GB RAM | DASLab 2.00-bit, 257 GB | about 4.7 (its mmap path 1.8-2.2) |
| Strata | RTX 3060 12 GB, 46 GB RAM | not stated | 8-10 |
| [Strata-DS](https://github.com/Jackwwg83/Strata-DS/blob/main/docs/DEEPSEEK_LAPTOP.md) (a Strata fork) | 24 GB laptop card, 64 GB RAM | SAGE 1.59-bit, about 120 GB | 30-39, with lossy skips |
| [zephel01](https://note.com/zephel01/n/n58f61f51ad18) | 32 GB card, 94 GB RAM | Unsloth UD-IQ1_M, after the Oct 28 revision, about 250 GB | 40.6 on 8 coding tasks (SWE-bench), with tail skip 4 and DSpark lookahead on; the card is AMD |

Unsloth's [guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash) says it runs on 64 GB of RAM, slower. It gives no tok/s.

## Credits

- DeepSeek for the model and Unsloth for the GGUF files ([Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF), [guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash)).
- [llama.cpp](https://github.com/ggml-org/llama.cpp), am17an for PR 31086, and danielhanchen, who wrote the first commit of PR 31086 (co-author on the merge) and opened the draft PR 30817 with that commit.
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (ikawrakow, who merged all of these): Zen 4 IQ1_S kernels (PR 212) and Smart Expert Reduction (PR 239). dmaivel wrote `--prefetch-experts` (PRs 2101, 2667). fijam wrote AVX-512 kernels for IQ1_S and IQ1_M (PR 2705) and the repacked IQ2 and IQ3 types (PR 2676), building on PR 2633 (ikawrakow, crediting fijam's patch).
- Nekotekina for llama.cpp PR 28055.
- [LLMario](https://github.com/mannysinghx/LLMario) (mannysinghx), PR 19.
- [Expert Sniper](https://huggingface.co/waltgrace/llama-cpp-expert-sniper) (waltgrace).
- [Strata](https://github.com/Niko1221/Strata) (Niko1221), 0.2.4 and 0.2.5, and its DeepSeek docs.
- [Strata-DS](https://github.com/Jackwwg83/Strata-DS) (Jackwwg83) and [moe-offload](https://github.com/Jakevin/moe-offload) (Jakevin) for test method ideas, and [llama.cpp discussion 28248](https://github.com/ggml-org/llama.cpp/discussions/28248).
- Also cited above: R3nzTheCodeGOD ([llama.cpp PR 31420](https://github.com/ggml-org/llama.cpp/pull/31420)), danveloper ([flash-moe](https://github.com/danveloper/flash-moe)), helgard-orlm ([repo](https://github.com/helgard-orlm/deepseek-v41-flash-8gb)), JigSawPT ([llama.cpp discussion 28766](https://github.com/ggml-org/llama.cpp/discussions/28766)), lifeidle ([repo](https://github.com/lifeidle/deepseek-v4.1-flash-5090-laptop-24gb-strata)), seppegadeyne ([post on Project Maya](https://dev.to/seppegadeyne/run-deepseek-v41-flash-748b-locally-with-project-maya-on-one-rtx-5090-4i4a)), zephel01 ([post](https://note.com/zephel01/n/n58f61f51ad18)) and [llama.cpp RFC 24528](https://github.com/ggml-org/llama.cpp/discussions/24528).
- Papers: [ZEDA](https://arxiv.org/html/2605.18643v2), [training-free halving](https://arxiv.org/abs/2609.04575), [LExI](https://arxiv.org/abs/2509.02753), [Alloc-MoE](https://arxiv.org/abs/2604.08133), [HybriMoE](https://arxiv.org/abs/2504.05897), [MoEpic](https://arxiv.org/abs/2509.08342).

## License

Text: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Copyright (c) 2026 the repository owner.

Independent write-up, not affiliated with or endorsed by DeepSeek, Unsloth, Anthropic or the llama.cpp project. No code yet; code later under MIT in a separate file. No model files are included.

## Questions for readers

1. Has anyone run ik_llama.cpp's `--prefetch-experts` on this model? I'd like to know how it does on a 64 GB PC.
2. Does anyone know why 16 threads hurts this much with the read hints on?
3. Has anyone compared the Oct 28 UD-IQ1_S files with the earlier ones, for speed or quality?
4. Anyone tried other layer picks for fewer experts on V4.1?
5. Has anyone overlapped the card part with the CPU part on a big MoE model in llama.cpp?
