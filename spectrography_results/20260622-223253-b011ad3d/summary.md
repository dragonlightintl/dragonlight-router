# Model Spectrography Report

- **Run ID:** 20260622-223253-b011ad3d
- **Started:** 2026-06-22T22:32:53.642247+00:00
- **Completed:** 2026-06-23T00:36:11.318401+00:00
- **Judge model:** nvidia_nim/qwen/qwen3.5-397b-a17b
- **Models evaluated:** 33
- **Total probe evaluations:** 315
- **Errors:** 126
- **Self-evaluations:** 0

## Model Rankings (Overall Average)

| Rank | Model | Avg Score |
|------|-------|-----------|
| 1 | nvidia_nim/meta/llama-3.1-70b-instruct | 0.8238 |
| 2 | nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | 0.7556 |
| 3 | nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | 0.6526 |
| 4 | nvidia_nim/deepseek-ai/deepseek-v4-flash | 0.6192 |
| 5 | nvidia_nim/z-ai/glm-5.1 | 0.6102 |
| 6 | nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | 0.6056 |
| 7 | nvidia_nim/mistralai/mistral-nemotron | 0.6026 |
| 8 | nvidia_nim/mistralai/mistral-small-4-119b-2603 | 0.6026 |
| 9 | nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | 0.5933 |
| 10 | nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | 0.5855 |
| 11 | nvidia_nim/minimaxai/minimax-m3 | 0.5851 |
| 12 | nvidia_nim/google/gemma-4-31b-it | 0.5718 |
| 13 | nvidia_nim/mistralai/ministral-14b-instruct-2512 | 0.5630 |
| 14 | nvidia_nim/mistralai/mistral-medium-3.5-128b | 0.5242 |
| 15 | nvidia_nim/openai/gpt-oss-120b | 0.5239 |
| 16 | gemini/gemini-2.5-flash-lite | 0.5183 |
| 17 | nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | 0.5145 |
| 18 | nvidia_nim/meta/llama-3.2-3b-instruct | 0.5037 |
| 19 | nvidia_nim/meta/llama-3.3-70b-instruct | 0.5023 |
| 20 | nvidia_nim/openai/gpt-oss-20b | 0.4871 |
| 21 | nvidia_nim/minimaxai/minimax-m2.7 | 0.4826 |
| 22 | nvidia_nim/meta/llama-3.1-8b-instruct | 0.4799 |
| 23 | nvidia_nim/google/gemma-3n-e4b-it | 0.4723 |
| 24 | nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | 0.4554 |
| 25 | nvidia_nim/stepfun-ai/step-3.5-flash | 0.4523 |
| 26 | nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | 0.3771 |
| 27 | nvidia_nim/google/gemma-2-2b-it | 0.3521 |
| 28 | gemini/gemma-4-31b-it | 0.3388 |
| 29 | nvidia_nim/sarvamai/sarvam-m | 0.3286 |
| 30 | nvidia_nim/nvidia/nemotron-3-super-120b-a12b | 0.3115 |
| 31 | nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | 0.2890 |
| 32 | gemini/gemma-4-26b-a4b-it | 0.2253 |
| 33 | nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | 0.1902 |

## Dimension Rankings

**domain/business:** nvidia_nim/z-ai/glm-5.1, nvidia_nim/qwen/qwen3-next-80b-a3b-instruct, nvidia_nim/meta/llama-3.2-3b-instruct, nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2
**domain/code:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/mistralai/mistral-small-4-119b-2603, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash
**domain/creative_writing:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/sarvamai/sarvam-m, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/mistralai/mistral-nemotron
**domain/general:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5, nvidia_nim/mistralai/mistral-small-4-119b-2603, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash
**domain/legal:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash
**domain/technical:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash
**qs/balanced:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5, nvidia_nim/mistralai/mistral-small-4-119b-2603, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash
**qs/quality:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/mistralai/mistral-nemotron, nvidia_nim/mistralai/mistral-medium-3.5-128b
**qs/speed:** nvidia_nim/z-ai/glm-5.1, nvidia_nim/qwen/qwen3-next-80b-a3b-instruct, nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512, nvidia_nim/mistralai/ministral-14b-instruct-2512, nvidia_nim/minimaxai/minimax-m2.7
**task/analysis:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash
**task/creative:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/sarvamai/sarvam-m, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512
**task/generation:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2, nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5, nvidia_nim/mistralai/mistral-small-4-119b-2603
**task/lookup:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512
**task/reasoning:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-31b-it, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/google/gemma-2-2b-it
**task/refactoring:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash
**task/summarization:** nvidia_nim/z-ai/glm-5.1, nvidia_nim/qwen/qwen3-next-80b-a3b-instruct, nvidia_nim/meta/llama-3.2-3b-instruct, nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2
**task/translation:** nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5, nvidia_nim/mistralai/mistral-small-4-119b-2603, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash

## Proficiencies & Deficiencies

**gemini/gemini-2.5-flash-lite**
- Top: task/reasoning (1.00), domain/general (0.84), qs/speed (0.79)
- Low: qs/quality (0.31), domain/business (0.24), task/lookup (0.13)

**gemini/gemma-4-26b-a4b-it**
- Top: task/analysis (0.50), task/refactoring (0.50), domain/creative_writing (0.50)
- Low: task/lookup (0.00), task/reasoning (0.00), task/translation (0.00)

**gemini/gemma-4-31b-it**
- Top: task/translation (0.69), qs/quality (0.69), task/analysis (0.50)
- Low: task/summarization (0.08), domain/business (0.08), task/lookup (0.07)

**nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct**
- Top: qs/quality (0.92), task/creative (0.90), domain/creative_writing (0.86)
- Low: domain/legal (0.50), domain/technical (0.50), task/translation (0.38)

**nvidia_nim/deepseek-ai/deepseek-v4-flash**
- Top: domain/general (0.88), qs/balanced (0.87), task/lookup (0.87)
- Low: qs/quality (0.35), task/creative (0.28), domain/creative_writing (0.23)

**nvidia_nim/google/gemma-2-2b-it**
- Top: task/creative (0.52), task/analysis (0.50), task/reasoning (0.50)
- Low: task/translation (0.19), domain/code (0.14), task/generation (0.10)

**nvidia_nim/google/gemma-3n-e4b-it**
- Top: qs/quality (0.73), domain/creative_writing (0.68), task/creative (0.66)
- Low: task/lookup (0.27), task/generation (0.24), domain/code (0.23)

**nvidia_nim/google/gemma-4-31b-it**
- Top: task/generation (0.97), domain/code (0.95), task/creative (0.79)
- Low: qs/speed (0.50), qs/quality (0.46), domain/creative_writing (0.36)

**nvidia_nim/meta/llama-3.1-70b-instruct**
- Top: task/creative (1.00), task/lookup (1.00), task/generation (1.00)
- Low: task/refactoring (0.50), domain/legal (0.50), domain/technical (0.50)

**nvidia_nim/meta/llama-3.1-8b-instruct**
- Top: domain/general (0.66), qs/balanced (0.65), task/translation (0.58)
- Low: task/lookup (0.33), task/generation (0.28), domain/code (0.27)

**nvidia_nim/meta/llama-3.2-3b-instruct**
- Top: domain/business (0.92), task/summarization (0.92), qs/speed (0.82)
- Low: domain/code (0.32), task/generation (0.31), task/translation (0.27)

**nvidia_nim/meta/llama-3.3-70b-instruct**
- Top: domain/general (0.81), qs/balanced (0.77), task/translation (0.73)
- Low: domain/code (0.36), task/generation (0.34), domain/creative_writing (0.27)

**nvidia_nim/meta/llama-4-maverick-17b-128e-instruct**
- Top: qs/quality (0.96), task/lookup (0.93), task/creative (0.93)
- Low: task/refactoring (0.50), domain/legal (0.50), domain/technical (0.50)

**nvidia_nim/minimaxai/minimax-m2.7**
- Top: qs/speed (0.86), task/lookup (0.53), task/summarization (0.50)
- Low: domain/general (0.41), task/creative (0.31), qs/balanced (0.10)

**nvidia_nim/minimaxai/minimax-m3**
- Top: task/creative (0.83), qs/quality (0.77), domain/general (0.75)
- Low: domain/technical (0.50), domain/code (0.41), task/generation (0.38)

**nvidia_nim/mistralai/ministral-14b-instruct-2512**
- Top: qs/speed (0.89), domain/general (0.69), qs/balanced (0.68)
- Low: domain/technical (0.50), domain/code (0.45), task/generation (0.41)

**nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512**
- Top: qs/speed (0.93), task/creative (0.86), qs/quality (0.81)
- Low: domain/code (0.50), task/generation (0.45), task/translation (0.42)

**nvidia_nim/mistralai/mistral-medium-3.5-128b**
- Top: qs/quality (0.85), qs/balanced (0.84), domain/general (0.78)
- Low: domain/business (0.28), task/summarization (0.25), qs/speed (0.21)

**nvidia_nim/mistralai/mistral-nemotron**
- Top: qs/quality (0.88), domain/creative_writing (0.82), task/translation (0.81)
- Low: domain/legal (0.50), domain/technical (0.50), qs/speed (0.29)

**nvidia_nim/mistralai/mistral-small-4-119b-2603**
- Top: domain/general (0.94), qs/balanced (0.94), task/translation (0.92)
- Low: qs/quality (0.27), task/creative (0.24), domain/creative_writing (0.18)

**nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1**
- Top: task/generation (0.55), task/lookup (0.50), task/analysis (0.50)
- Low: task/creative (0.21), domain/creative_writing (0.14), qs/quality (0.12)

**nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1**
- Top: task/creative (0.62), domain/business (0.60), domain/code (0.59)
- Low: task/generation (0.21), qs/balanced (0.19), domain/general (0.16)

**nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5**
- Top: domain/general (0.97), qs/balanced (0.97), task/translation (0.96)
- Low: domain/code (0.50), qs/quality (0.50), qs/speed (0.50)

**nvidia_nim/nvidia/nemotron-3-nano-30b-a3b**
- Top: task/lookup (0.50), task/analysis (0.50), task/reasoning (0.50)
- Low: qs/quality (0.04), task/creative (0.00), domain/creative_writing (0.00)

**nvidia_nim/nvidia/nemotron-3-super-120b-a12b**
- Top: domain/code (0.64), task/generation (0.59), task/lookup (0.50)
- Low: domain/creative_writing (0.09), task/creative (0.07), task/translation (0.04)

**nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b**
- Top: task/lookup (0.50), task/analysis (0.50), task/reasoning (0.50)
- Low: qs/quality (0.00), qs/balanced (0.00), qs/speed (0.00)

**nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2**
- Top: task/generation (0.93), domain/business (0.84), task/summarization (0.83)
- Low: qs/balanced (0.35), domain/general (0.28), task/translation (0.08)

**nvidia_nim/openai/gpt-oss-120b**
- Top: qs/balanced (0.71), task/generation (0.62), domain/general (0.59)
- Low: qs/quality (0.50), qs/speed (0.50), task/creative (0.48)

**nvidia_nim/openai/gpt-oss-20b**
- Top: task/generation (0.66), qs/balanced (0.52), task/summarization (0.50)
- Low: qs/speed (0.50), domain/general (0.44), task/creative (0.17)

**nvidia_nim/qwen/qwen3-next-80b-a3b-instruct**
- Top: qs/speed (0.96), domain/business (0.96), task/summarization (0.96)
- Low: qs/balanced (0.32), domain/general (0.25), task/translation (0.23)

**nvidia_nim/sarvamai/sarvam-m**
- Top: task/creative (0.97), domain/creative_writing (0.95), task/lookup (0.50)
- Low: task/summarization (0.04), domain/business (0.04), qs/speed (0.04)

**nvidia_nim/stepfun-ai/step-3.5-flash**
- Top: domain/business (0.64), task/summarization (0.62), task/lookup (0.50)
- Low: task/creative (0.34), qs/balanced (0.13), domain/general (0.09)

**nvidia_nim/z-ai/glm-5.1**
- Top: task/summarization (1.00), domain/business (1.00), qs/speed (1.00)
- Low: task/translation (0.46), qs/balanced (0.26), domain/general (0.19)

## Calibration Deltas

| Model | Dimension | Delta |
|-------|-----------|-------|
| gemini/gemini-2.5-flash-lite | domain/business | +0.2400 |
| gemini/gemini-2.5-flash-lite | domain/code | +0.5000 |
| gemini/gemini-2.5-flash-lite | domain/creative_writing | +0.0000 |
| gemini/gemini-2.5-flash-lite | domain/general | +0.3438 |
| gemini/gemini-2.5-flash-lite | domain/legal | +0.1667 |
| gemini/gemini-2.5-flash-lite | domain/technical | +0.3000 |
| gemini/gemini-2.5-flash-lite | qs/balanced | +0.5000 |
| gemini/gemini-2.5-flash-lite | qs/quality | +0.0923 |
| gemini/gemini-2.5-flash-lite | qs/speed | +0.2143 |
| gemini/gemini-2.5-flash-lite | task/analysis | +0.1000 |
| gemini/gemini-2.5-flash-lite | task/creative | +0.0000 |
| gemini/gemini-2.5-flash-lite | task/generation | +0.5000 |
| gemini/gemini-2.5-flash-lite | task/lookup | +0.3667 |
| gemini/gemini-2.5-flash-lite | task/reasoning | +0.6000 |
| gemini/gemini-2.5-flash-lite | task/refactoring | +0.2500 |
| gemini/gemini-2.5-flash-lite | task/summarization | +0.5000 |
| gemini/gemini-2.5-flash-lite | task/translation | +0.0000 |
| gemini/gemma-4-26b-a4b-it | domain/business | +0.1333 |
| gemini/gemma-4-26b-a4b-it | domain/code | +0.1000 |
| gemini/gemma-4-26b-a4b-it | domain/creative_writing | +0.5000 |
| gemini/gemma-4-26b-a4b-it | domain/general | +0.4688 |
| gemini/gemma-4-26b-a4b-it | domain/legal | +0.1667 |
| gemini/gemma-4-26b-a4b-it | domain/technical | +0.1000 |
| gemini/gemma-4-26b-a4b-it | qs/balanced | +0.5677 |
| gemini/gemma-4-26b-a4b-it | qs/quality | +0.7231 |
| gemini/gemma-4-26b-a4b-it | qs/speed | +0.1071 |
| gemini/gemma-4-26b-a4b-it | task/analysis | +0.3000 |
| gemini/gemma-4-26b-a4b-it | task/creative | +0.8966 |
| gemini/gemma-4-26b-a4b-it | task/generation | +0.2155 |
| gemini/gemma-4-26b-a4b-it | task/lookup | +0.5000 |
| gemini/gemma-4-26b-a4b-it | task/reasoning | +1.0000 |
| gemini/gemma-4-26b-a4b-it | task/refactoring | +0.5000 |
| gemini/gemma-4-26b-a4b-it | task/summarization | +0.2917 |
| gemini/gemma-4-26b-a4b-it | task/translation | +0.5000 |
| gemini/gemma-4-31b-it | domain/business | +0.9200 |
| gemini/gemma-4-31b-it | domain/code | +0.6182 |
| gemini/gemma-4-31b-it | domain/creative_writing | +0.1667 |
| gemini/gemma-4-31b-it | domain/general | +0.2812 |
| gemini/gemma-4-31b-it | domain/legal | +0.5000 |
| gemini/gemma-4-31b-it | domain/technical | +0.1000 |
| gemini/gemma-4-31b-it | qs/balanced | +0.5097 |
| gemini/gemma-4-31b-it | qs/quality | +0.0923 |
| gemini/gemma-4-31b-it | qs/speed | +0.3214 |
| gemini/gemma-4-31b-it | task/analysis | +0.5000 |
| gemini/gemma-4-31b-it | task/creative | +0.5288 |
| gemini/gemma-4-31b-it | task/generation | +0.6121 |
| gemini/gemma-4-31b-it | task/lookup | +0.4333 |
| gemini/gemma-4-31b-it | task/reasoning | +0.1000 |
| gemini/gemma-4-31b-it | task/refactoring | +0.2500 |
| gemini/gemma-4-31b-it | task/summarization | +0.6667 |
| gemini/gemma-4-31b-it | task/translation | +0.1923 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/business | +0.1800 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/code | +0.4289 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/creative_writing | +0.5944 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/general | +0.1250 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/legal | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/technical | +0.1875 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | qs/balanced | +0.0806 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | qs/quality | +0.2898 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | qs/speed | +0.1429 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/analysis | +0.1154 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/creative | +0.6274 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/generation | +0.3836 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/lookup | +0.3000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/reasoning | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/refactoring | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/summarization | +0.1667 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/translation | +0.1154 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/business | +0.2200 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/code | +0.0057 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/creative_writing | +0.0804 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/general | +0.3750 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/legal | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/technical | +0.1562 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | qs/balanced | +0.3710 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | qs/quality | +0.5205 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | qs/speed | +0.2648 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/analysis | +0.2308 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/creative | +0.0318 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/generation | +0.1681 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/lookup | +0.3667 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/reasoning | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/refactoring | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/summarization | +0.4490 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/translation | +0.3462 |
| nvidia_nim/google/gemma-2-2b-it | domain/business | +0.1400 |
| nvidia_nim/google/gemma-2-2b-it | domain/code | +0.0198 |
| nvidia_nim/google/gemma-2-2b-it | domain/creative_writing | +0.0245 |
| nvidia_nim/google/gemma-2-2b-it | domain/general | +0.1875 |
| nvidia_nim/google/gemma-2-2b-it | domain/legal | +0.0000 |
| nvidia_nim/google/gemma-2-2b-it | domain/technical | +0.3438 |
| nvidia_nim/google/gemma-2-2b-it | qs/balanced | +0.2090 |
| nvidia_nim/google/gemma-2-2b-it | qs/quality | +0.0975 |
| nvidia_nim/google/gemma-2-2b-it | qs/speed | +0.2574 |
| nvidia_nim/google/gemma-2-2b-it | task/analysis | +0.4231 |
| nvidia_nim/google/gemma-2-2b-it | task/creative | +0.1326 |
| nvidia_nim/google/gemma-2-2b-it | task/generation | +0.0216 |
| nvidia_nim/google/gemma-2-2b-it | task/lookup | +0.3000 |
| nvidia_nim/google/gemma-2-2b-it | task/reasoning | +0.0000 |
| nvidia_nim/google/gemma-2-2b-it | task/refactoring | +0.0000 |
| nvidia_nim/google/gemma-2-2b-it | task/summarization | +0.2592 |
| nvidia_nim/google/gemma-2-2b-it | task/translation | +0.3077 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/business | +0.1000 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/code | +0.0539 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/creative_writing | +0.2587 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/general | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/legal | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/technical | +0.2812 |
| nvidia_nim/google/gemma-3n-e4b-it | qs/balanced | +0.1473 |
| nvidia_nim/google/gemma-3n-e4b-it | qs/quality | +0.2641 |
| nvidia_nim/google/gemma-3n-e4b-it | qs/speed | +0.2241 |
| nvidia_nim/google/gemma-3n-e4b-it | task/analysis | +0.1154 |
| nvidia_nim/google/gemma-3n-e4b-it | task/creative | +0.2321 |
| nvidia_nim/google/gemma-3n-e4b-it | task/generation | +0.2274 |
| nvidia_nim/google/gemma-3n-e4b-it | task/lookup | +0.2333 |
| nvidia_nim/google/gemma-3n-e4b-it | task/reasoning | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | task/refactoring | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | task/summarization | +0.2639 |
| nvidia_nim/google/gemma-3n-e4b-it | task/translation | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | domain/business | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | domain/code | +0.0143 |
| nvidia_nim/google/gemma-4-31b-it | domain/creative_writing | +0.4826 |
| nvidia_nim/google/gemma-4-31b-it | domain/general | +0.0312 |
| nvidia_nim/google/gemma-4-31b-it | domain/legal | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | domain/technical | +0.4375 |
| nvidia_nim/google/gemma-4-31b-it | qs/balanced | +0.3436 |
| nvidia_nim/google/gemma-4-31b-it | qs/quality | +0.5052 |
| nvidia_nim/google/gemma-4-31b-it | qs/speed | +0.4655 |
| nvidia_nim/google/gemma-4-31b-it | task/analysis | +0.4615 |
| nvidia_nim/google/gemma-4-31b-it | task/creative | +0.0531 |
| nvidia_nim/google/gemma-4-31b-it | task/generation | +0.0593 |
| nvidia_nim/google/gemma-4-31b-it | task/lookup | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | task/reasoning | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | task/refactoring | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | task/summarization | +0.4630 |
| nvidia_nim/google/gemma-4-31b-it | task/translation | +0.0385 |
| nvidia_nim/meta/llama-3.1-70b-instruct | domain/business | +0.3800 |
| nvidia_nim/meta/llama-3.1-70b-instruct | domain/code | +0.2500 |
| nvidia_nim/meta/llama-3.1-70b-instruct | domain/creative_writing | +0.1154 |
| nvidia_nim/meta/llama-3.1-70b-instruct | domain/general | +0.5000 |
| nvidia_nim/meta/llama-3.1-70b-instruct | domain/legal | +0.0000 |
| nvidia_nim/meta/llama-3.1-70b-instruct | domain/technical | +0.2500 |
| nvidia_nim/meta/llama-3.1-70b-instruct | qs/balanced | +0.5000 |
| nvidia_nim/meta/llama-3.1-70b-instruct | qs/quality | +0.2333 |
| nvidia_nim/meta/llama-3.1-70b-instruct | qs/speed | +0.4397 |
| nvidia_nim/meta/llama-3.1-70b-instruct | task/analysis | +0.1538 |
| nvidia_nim/meta/llama-3.1-70b-instruct | task/creative | +0.1154 |
| nvidia_nim/meta/llama-3.1-70b-instruct | task/generation | +0.0625 |
| nvidia_nim/meta/llama-3.1-70b-instruct | task/lookup | +0.5000 |
| nvidia_nim/meta/llama-3.1-70b-instruct | task/reasoning | +0.0000 |
| nvidia_nim/meta/llama-3.1-70b-instruct | task/refactoring | +0.0000 |
| nvidia_nim/meta/llama-3.1-70b-instruct | task/summarization | +0.7269 |
| nvidia_nim/meta/llama-3.1-70b-instruct | task/translation | +0.5000 |
| nvidia_nim/meta/llama-3.1-8b-instruct | domain/business | +0.0600 |
| nvidia_nim/meta/llama-3.1-8b-instruct | domain/code | +0.3523 |
| nvidia_nim/meta/llama-3.1-8b-instruct | domain/creative_writing | +0.4686 |
| nvidia_nim/meta/llama-3.1-8b-instruct | domain/general | +0.1562 |
| nvidia_nim/meta/llama-3.1-8b-instruct | domain/legal | +0.0000 |
| nvidia_nim/meta/llama-3.1-8b-instruct | domain/technical | +0.3125 |
| nvidia_nim/meta/llama-3.1-8b-instruct | qs/balanced | +0.1452 |
| nvidia_nim/meta/llama-3.1-8b-instruct | qs/quality | +0.1667 |
| nvidia_nim/meta/llama-3.1-8b-instruct | qs/speed | +0.1909 |
| nvidia_nim/meta/llama-3.1-8b-instruct | task/analysis | +0.0769 |
| nvidia_nim/meta/llama-3.1-8b-instruct | task/creative | +0.3714 |
| nvidia_nim/meta/llama-3.1-8b-instruct | task/generation | +0.1821 |
| nvidia_nim/meta/llama-3.1-8b-instruct | task/lookup | +0.1667 |
| nvidia_nim/meta/llama-3.1-8b-instruct | task/reasoning | +0.0000 |
| nvidia_nim/meta/llama-3.1-8b-instruct | task/refactoring | +0.0000 |
| nvidia_nim/meta/llama-3.1-8b-instruct | task/summarization | +0.2315 |
| nvidia_nim/meta/llama-3.1-8b-instruct | task/translation | +0.0769 |
| nvidia_nim/meta/llama-3.2-3b-instruct | domain/business | +0.4200 |
| nvidia_nim/meta/llama-3.2-3b-instruct | domain/code | +0.2244 |
| nvidia_nim/meta/llama-3.2-3b-instruct | domain/creative_writing | +0.6433 |
| nvidia_nim/meta/llama-3.2-3b-instruct | domain/general | +0.0312 |
| nvidia_nim/meta/llama-3.2-3b-instruct | domain/legal | +0.0000 |
| nvidia_nim/meta/llama-3.2-3b-instruct | domain/technical | +0.2188 |
| nvidia_nim/meta/llama-3.2-3b-instruct | qs/balanced | +0.2230 |
| nvidia_nim/meta/llama-3.2-3b-instruct | qs/quality | +0.0102 |
| nvidia_nim/meta/llama-3.2-3b-instruct | qs/speed | +0.4421 |
| nvidia_nim/meta/llama-3.2-3b-instruct | task/analysis | +0.2692 |
| nvidia_nim/meta/llama-3.2-3b-instruct | task/creative | +0.5477 |
| nvidia_nim/meta/llama-3.2-3b-instruct | task/generation | +0.0022 |
| nvidia_nim/meta/llama-3.2-3b-instruct | task/lookup | +0.1000 |
| nvidia_nim/meta/llama-3.2-3b-instruct | task/reasoning | +0.0000 |
| nvidia_nim/meta/llama-3.2-3b-instruct | task/refactoring | +0.0000 |
| nvidia_nim/meta/llama-3.2-3b-instruct | task/summarization | +0.6945 |
| nvidia_nim/meta/llama-3.2-3b-instruct | task/translation | +0.2308 |
| nvidia_nim/meta/llama-3.3-70b-instruct | domain/business | +0.0200 |
| nvidia_nim/meta/llama-3.3-70b-instruct | domain/code | +0.0739 |
| nvidia_nim/meta/llama-3.3-70b-instruct | domain/creative_writing | +0.2273 |
| nvidia_nim/meta/llama-3.3-70b-instruct | domain/general | +0.3125 |
| nvidia_nim/meta/llama-3.3-70b-instruct | domain/legal | +0.0000 |
| nvidia_nim/meta/llama-3.3-70b-instruct | domain/technical | +0.0938 |
| nvidia_nim/meta/llama-3.3-70b-instruct | qs/balanced | +0.1655 |
| nvidia_nim/meta/llama-3.3-70b-instruct | qs/quality | +0.0154 |
| nvidia_nim/meta/llama-3.3-70b-instruct | qs/speed | +0.0714 |
| nvidia_nim/meta/llama-3.3-70b-instruct | task/analysis | +0.0385 |
| nvidia_nim/meta/llama-3.3-70b-instruct | task/creative | +0.1207 |
| nvidia_nim/meta/llama-3.3-70b-instruct | task/generation | +0.0927 |
| nvidia_nim/meta/llama-3.3-70b-instruct | task/lookup | +0.0333 |
| nvidia_nim/meta/llama-3.3-70b-instruct | task/reasoning | +0.0000 |
| nvidia_nim/meta/llama-3.3-70b-instruct | task/refactoring | +0.0000 |
| nvidia_nim/meta/llama-3.3-70b-instruct | task/summarization | +0.0417 |
| nvidia_nim/meta/llama-3.3-70b-instruct | task/translation | +0.2308 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | domain/business | +0.2600 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | domain/code | +0.1364 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | domain/creative_writing | +0.0909 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | domain/general | +0.4062 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | domain/legal | +0.0000 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | domain/technical | +0.4688 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | qs/balanced | +0.0968 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | qs/quality | +0.0385 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | qs/speed | +0.2143 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | task/analysis | +0.5000 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | task/creative | +0.0690 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | task/generation | +0.1412 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | task/lookup | +0.4333 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | task/reasoning | +0.0000 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | task/refactoring | +0.0000 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | task/summarization | +0.2500 |
| nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | task/translation | +0.3846 |
| nvidia_nim/minimaxai/minimax-m2.7 | domain/business | +0.0000 |
| nvidia_nim/minimaxai/minimax-m2.7 | domain/code | +0.4688 |
| nvidia_nim/minimaxai/minimax-m2.7 | domain/creative_writing | +0.0000 |
| nvidia_nim/minimaxai/minimax-m2.7 | domain/general | +0.0938 |
| nvidia_nim/minimaxai/minimax-m2.7 | domain/legal | +0.0000 |
| nvidia_nim/minimaxai/minimax-m2.7 | domain/technical | +0.0938 |
| nvidia_nim/minimaxai/minimax-m2.7 | qs/balanced | +0.4032 |
| nvidia_nim/minimaxai/minimax-m2.7 | qs/quality | +0.0000 |
| nvidia_nim/minimaxai/minimax-m2.7 | qs/speed | +0.7192 |
| nvidia_nim/minimaxai/minimax-m2.7 | task/analysis | +0.0000 |
| nvidia_nim/minimaxai/minimax-m2.7 | task/creative | +0.1897 |
| nvidia_nim/minimaxai/minimax-m2.7 | task/generation | +0.4688 |
| nvidia_nim/minimaxai/minimax-m2.7 | task/lookup | +0.0333 |
| nvidia_nim/minimaxai/minimax-m2.7 | task/reasoning | +0.0000 |
| nvidia_nim/minimaxai/minimax-m2.7 | task/refactoring | +0.0000 |
| nvidia_nim/minimaxai/minimax-m2.7 | task/summarization | +0.0185 |
| nvidia_nim/minimaxai/minimax-m2.7 | task/translation | +0.0000 |
| nvidia_nim/minimaxai/minimax-m3 | domain/business | +0.0200 |
| nvidia_nim/minimaxai/minimax-m3 | domain/code | +0.0909 |
| nvidia_nim/minimaxai/minimax-m3 | domain/creative_writing | +0.2658 |
| nvidia_nim/minimaxai/minimax-m3 | domain/general | +0.2500 |
| nvidia_nim/minimaxai/minimax-m3 | domain/legal | +0.0000 |
| nvidia_nim/minimaxai/minimax-m3 | domain/technical | +0.0625 |
| nvidia_nim/minimaxai/minimax-m3 | qs/balanced | +0.1767 |
| nvidia_nim/minimaxai/minimax-m3 | qs/quality | +0.2025 |
| nvidia_nim/minimaxai/minimax-m3 | qs/speed | +0.1243 |
| nvidia_nim/minimaxai/minimax-m3 | task/analysis | +0.3846 |
| nvidia_nim/minimaxai/minimax-m3 | task/creative | +0.3661 |
| nvidia_nim/minimaxai/minimax-m3 | task/generation | +0.1832 |
| nvidia_nim/minimaxai/minimax-m3 | task/lookup | +0.1000 |
| nvidia_nim/minimaxai/minimax-m3 | task/reasoning | +0.0000 |
| nvidia_nim/minimaxai/minimax-m3 | task/refactoring | +0.5000 |
| nvidia_nim/minimaxai/minimax-m3 | task/summarization | +0.1296 |
| nvidia_nim/minimaxai/minimax-m3 | task/translation | +0.1154 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | domain/business | +0.0000 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | domain/code | +0.0143 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | domain/creative_writing | +0.0769 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | domain/general | +0.1875 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | domain/legal | +0.0000 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | domain/technical | +0.1250 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | qs/balanced | +0.2861 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | qs/quality | +0.1615 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | qs/speed | +0.2722 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | task/analysis | +0.0385 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | task/creative | +0.0093 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | task/generation | +0.0862 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | task/lookup | +0.1667 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | task/reasoning | +0.0000 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | task/refactoring | +0.0000 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | task/summarization | +0.0556 |
| nvidia_nim/mistralai/ministral-14b-instruct-2512 | task/translation | +0.1538 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | domain/business | +0.0000 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | domain/code | +0.0625 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | domain/creative_writing | +0.1573 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | domain/general | +0.0625 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | domain/legal | +0.0000 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | domain/technical | +0.1562 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | qs/balanced | +0.1907 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | qs/quality | +0.3077 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | qs/speed | +0.2734 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | task/analysis | +0.1923 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | task/creative | +0.2467 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | task/generation | +0.3017 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | task/lookup | +0.2333 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | task/reasoning | +0.0000 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | task/refactoring | +0.0000 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | task/summarization | +0.0926 |
| nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | task/translation | +0.0769 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | domain/business | +0.2200 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | domain/code | +0.3438 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | domain/creative_writing | +0.1538 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | domain/general | +0.2812 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | domain/legal | +0.0000 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | domain/technical | +0.1875 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | qs/balanced | +0.0561 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | qs/quality | +0.0538 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | qs/speed | +0.4754 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | task/analysis | +0.3846 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | task/creative | +0.2055 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | task/generation | +0.2984 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | task/lookup | +0.0000 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | task/reasoning | +0.0000 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | task/refactoring | +0.0000 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | task/summarization | +0.3796 |
| nvidia_nim/mistralai/mistral-medium-3.5-128b | task/translation | +0.2692 |
| nvidia_nim/mistralai/mistral-nemotron | domain/business | +0.0600 |
| nvidia_nim/mistralai/mistral-nemotron | domain/code | +0.1420 |
| nvidia_nim/mistralai/mistral-nemotron | domain/creative_writing | +0.3182 |
| nvidia_nim/mistralai/mistral-nemotron | domain/general | +0.2188 |
| nvidia_nim/mistralai/mistral-nemotron | domain/legal | +0.0000 |
| nvidia_nim/mistralai/mistral-nemotron | domain/technical | +0.0312 |
| nvidia_nim/mistralai/mistral-nemotron | qs/balanced | +0.1543 |
| nvidia_nim/mistralai/mistral-nemotron | qs/quality | +0.2846 |
| nvidia_nim/mistralai/mistral-nemotron | qs/speed | +0.2315 |
| nvidia_nim/mistralai/mistral-nemotron | task/analysis | +0.2692 |
| nvidia_nim/mistralai/mistral-nemotron | task/creative | +0.2586 |
| nvidia_nim/mistralai/mistral-nemotron | task/generation | +0.1390 |
| nvidia_nim/mistralai/mistral-nemotron | task/lookup | +0.0000 |
| nvidia_nim/mistralai/mistral-nemotron | task/reasoning | +0.0000 |
| nvidia_nim/mistralai/mistral-nemotron | task/refactoring | +0.2500 |
| nvidia_nim/mistralai/mistral-nemotron | task/summarization | +0.1343 |
| nvidia_nim/mistralai/mistral-nemotron | task/translation | +0.3077 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | domain/business | +0.3000 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | domain/code | +0.1279 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | domain/creative_writing | +0.3182 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | domain/general | +0.4375 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | domain/legal | +0.0000 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | domain/technical | +0.0000 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | qs/balanced | +0.4355 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | qs/quality | +0.5308 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | qs/speed | +0.1588 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | task/analysis | +0.3077 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | task/creative | +0.2586 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | task/generation | +0.1746 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | task/lookup | +0.0000 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | task/reasoning | +0.0000 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | task/refactoring | +0.0000 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | task/summarization | +0.3473 |
| nvidia_nim/mistralai/mistral-small-4-119b-2603 | task/translation | +0.4231 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | domain/business | +0.1800 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | domain/code | +0.3750 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | domain/creative_writing | +0.0559 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | domain/general | +0.1562 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | domain/legal | +0.0000 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | domain/technical | +0.0000 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | qs/balanced | +0.4390 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | qs/quality | +0.0821 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | qs/speed | +0.2500 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | task/analysis | +0.4615 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | task/creative | +0.0146 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | task/generation | +0.3955 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | task/lookup | +0.0000 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | task/reasoning | +0.0000 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | task/refactoring | +0.0000 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | task/summarization | +0.2083 |
| nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | task/translation | +0.1923 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | domain/business | +0.1000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | domain/code | +0.3721 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | domain/creative_writing | +0.1853 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | domain/general | +0.3438 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | domain/legal | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | domain/technical | +0.2188 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | qs/balanced | +0.0196 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | qs/quality | +0.0436 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | qs/speed | +0.0800 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | task/analysis | +0.1538 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | task/creative | +0.1101 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | task/generation | +0.0119 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | task/lookup | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | task/reasoning | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | task/refactoring | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | task/summarization | +0.1204 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | task/translation | +0.1538 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | domain/business | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | domain/code | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | domain/creative_writing | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | domain/general | +0.4688 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | domain/legal | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | domain/technical | +0.5000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | qs/balanced | +0.4677 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | qs/quality | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | qs/speed | +0.4655 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | task/analysis | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | task/creative | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | task/generation | +0.3966 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | task/lookup | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | task/reasoning | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | task/refactoring | +0.0000 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | task/summarization | +0.4630 |
| nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | task/translation | +0.4615 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | domain/business | +0.3800 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | domain/code | +0.0966 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | domain/creative_writing | +0.0385 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | domain/general | +0.1250 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | domain/legal | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | domain/technical | +0.4688 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | qs/balanced | +0.3324 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | qs/quality | +0.0282 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | qs/speed | +0.0320 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | task/analysis | +0.3462 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | task/creative | +0.0385 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | task/generation | +0.0151 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | task/lookup | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | task/reasoning | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | task/refactoring | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | task/summarization | +0.1250 |
| nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | task/translation | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | domain/business | +0.3400 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | domain/code | +0.2386 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | domain/creative_writing | +0.6783 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | domain/general | +0.3750 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | domain/legal | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | domain/technical | +0.3750 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | qs/balanced | +0.7083 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | qs/quality | +0.1462 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | qs/speed | +0.6170 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | task/analysis | +0.4231 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | task/creative | +0.7002 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | task/generation | +0.2424 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | task/lookup | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | task/reasoning | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | task/refactoring | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | task/summarization | +0.5740 |
| nvidia_nim/nvidia/nemotron-3-super-120b-a12b | task/translation | +0.4615 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | domain/business | +0.5000 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | domain/code | +0.3125 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | domain/creative_writing | +0.0455 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | domain/general | +0.5000 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | domain/legal | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | domain/technical | +0.2500 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | qs/balanced | +0.4783 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | qs/quality | +0.1667 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | qs/speed | +0.7586 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | task/analysis | +0.3077 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | task/creative | +0.0345 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | task/generation | +0.5938 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | task/lookup | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | task/reasoning | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | task/refactoring | +0.0000 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | task/summarization | +0.7778 |
| nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | task/translation | +0.3462 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | domain/business | +0.3400 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | domain/code | +0.4062 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | domain/creative_writing | +0.2692 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | domain/general | +0.2188 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | domain/legal | +0.0000 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | domain/technical | +0.2812 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | qs/balanced | +0.1452 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | qs/quality | +0.3000 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | qs/speed | +0.3645 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | task/analysis | +0.0000 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | task/creative | +0.2692 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | task/generation | +0.1185 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | task/lookup | +0.0000 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | task/reasoning | +0.0000 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | task/refactoring | +0.0000 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | task/summarization | +0.0185 |
| nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | task/translation | +0.4231 |
| nvidia_nim/openai/gpt-oss-120b | domain/business | +0.0000 |
| nvidia_nim/openai/gpt-oss-120b | domain/code | +0.2500 |
| nvidia_nim/openai/gpt-oss-120b | domain/creative_writing | +0.3846 |
| nvidia_nim/openai/gpt-oss-120b | domain/general | +0.0938 |
| nvidia_nim/openai/gpt-oss-120b | domain/legal | +0.0000 |
| nvidia_nim/openai/gpt-oss-120b | domain/technical | +0.3125 |
| nvidia_nim/openai/gpt-oss-120b | qs/balanced | +0.6662 |
| nvidia_nim/openai/gpt-oss-120b | qs/quality | +0.2667 |
| nvidia_nim/openai/gpt-oss-120b | qs/speed | +0.3276 |
| nvidia_nim/openai/gpt-oss-120b | task/analysis | +0.0000 |
| nvidia_nim/openai/gpt-oss-120b | task/creative | +0.3674 |
| nvidia_nim/openai/gpt-oss-120b | task/generation | +0.3707 |
| nvidia_nim/openai/gpt-oss-120b | task/lookup | +0.0000 |
| nvidia_nim/openai/gpt-oss-120b | task/reasoning | +0.0000 |
| nvidia_nim/openai/gpt-oss-120b | task/refactoring | +0.0000 |
| nvidia_nim/openai/gpt-oss-120b | task/summarization | +0.3519 |
| nvidia_nim/openai/gpt-oss-120b | task/translation | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | domain/business | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | domain/code | +0.0938 |
| nvidia_nim/openai/gpt-oss-20b | domain/creative_writing | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | domain/general | +0.0625 |
| nvidia_nim/openai/gpt-oss-20b | domain/legal | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | domain/technical | +0.3438 |
| nvidia_nim/openai/gpt-oss-20b | qs/balanced | +0.2987 |
| nvidia_nim/openai/gpt-oss-20b | qs/quality | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | qs/speed | +0.3621 |
| nvidia_nim/openai/gpt-oss-20b | task/analysis | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | task/creative | +0.3276 |
| nvidia_nim/openai/gpt-oss-20b | task/generation | +0.3740 |
| nvidia_nim/openai/gpt-oss-20b | task/lookup | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | task/reasoning | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | task/refactoring | +0.0000 |
| nvidia_nim/openai/gpt-oss-20b | task/summarization | +0.3889 |
| nvidia_nim/openai/gpt-oss-20b | task/translation | +0.0000 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | domain/business | +0.4600 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | domain/code | +0.0880 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | domain/creative_writing | +0.0909 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | domain/general | +0.2500 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | domain/legal | +0.0000 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | domain/technical | +0.3750 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | qs/balanced | +0.5904 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | qs/quality | +0.2487 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | qs/speed | +0.0677 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | task/analysis | +0.0769 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | task/creative | +0.1897 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | task/generation | +0.1541 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | task/lookup | +0.0000 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | task/reasoning | +0.0000 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | task/refactoring | +0.0000 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | task/summarization | +0.4583 |
| nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | task/translation | +0.2692 |
| nvidia_nim/sarvamai/sarvam-m | domain/business | +0.4600 |
| nvidia_nim/sarvamai/sarvam-m | domain/code | +0.0170 |
| nvidia_nim/sarvamai/sarvam-m | domain/creative_writing | +0.8007 |
| nvidia_nim/sarvamai/sarvam-m | domain/general | +0.4375 |
| nvidia_nim/sarvamai/sarvam-m | domain/legal | +0.0000 |
| nvidia_nim/sarvamai/sarvam-m | domain/technical | +0.4375 |
| nvidia_nim/sarvamai/sarvam-m | qs/balanced | +0.0659 |
| nvidia_nim/sarvamai/sarvam-m | qs/quality | +0.1923 |
| nvidia_nim/sarvamai/sarvam-m | qs/speed | +0.1367 |
| nvidia_nim/sarvamai/sarvam-m | task/analysis | +0.2308 |
| nvidia_nim/sarvamai/sarvam-m | task/creative | +0.8117 |
| nvidia_nim/sarvamai/sarvam-m | task/generation | +0.0065 |
| nvidia_nim/sarvamai/sarvam-m | task/lookup | +0.0000 |
| nvidia_nim/sarvamai/sarvam-m | task/reasoning | +0.0000 |
| nvidia_nim/sarvamai/sarvam-m | task/refactoring | +0.0000 |
| nvidia_nim/sarvamai/sarvam-m | task/summarization | +0.2916 |
| nvidia_nim/sarvamai/sarvam-m | task/translation | +0.3846 |
| nvidia_nim/stepfun-ai/step-3.5-flash | domain/business | +0.1400 |
| nvidia_nim/stepfun-ai/step-3.5-flash | domain/code | +0.4375 |
| nvidia_nim/stepfun-ai/step-3.5-flash | domain/creative_writing | +0.3077 |
| nvidia_nim/stepfun-ai/step-3.5-flash | domain/general | +0.4062 |
| nvidia_nim/stepfun-ai/step-3.5-flash | domain/legal | +0.0000 |
| nvidia_nim/stepfun-ai/step-3.5-flash | domain/technical | +0.4062 |
| nvidia_nim/stepfun-ai/step-3.5-flash | qs/balanced | +0.3710 |
| nvidia_nim/stepfun-ai/step-3.5-flash | qs/quality | +0.4333 |
| nvidia_nim/stepfun-ai/step-3.5-flash | qs/speed | +0.5739 |
| nvidia_nim/stepfun-ai/step-3.5-flash | task/analysis | +0.0000 |
| nvidia_nim/stepfun-ai/step-3.5-flash | task/creative | +0.4629 |
| nvidia_nim/stepfun-ai/step-3.5-flash | task/generation | +0.3750 |
| nvidia_nim/stepfun-ai/step-3.5-flash | task/lookup | +0.0000 |
| nvidia_nim/stepfun-ai/step-3.5-flash | task/reasoning | +0.0000 |
| nvidia_nim/stepfun-ai/step-3.5-flash | task/refactoring | +0.0000 |
| nvidia_nim/stepfun-ai/step-3.5-flash | task/summarization | +0.3009 |
| nvidia_nim/stepfun-ai/step-3.5-flash | task/translation | +0.0000 |
| nvidia_nim/z-ai/glm-5.1 | domain/business | +0.5000 |
| nvidia_nim/z-ai/glm-5.1 | domain/code | +0.0711 |
| nvidia_nim/z-ai/glm-5.1 | domain/creative_writing | +0.0979 |
| nvidia_nim/z-ai/glm-5.1 | domain/general | +0.3125 |
| nvidia_nim/z-ai/glm-5.1 | domain/legal | +0.0000 |
| nvidia_nim/z-ai/glm-5.1 | domain/technical | +0.0625 |
| nvidia_nim/z-ai/glm-5.1 | qs/balanced | +0.4376 |
| nvidia_nim/z-ai/glm-5.1 | qs/quality | +0.0129 |
| nvidia_nim/z-ai/glm-5.1 | qs/speed | +0.4138 |
| nvidia_nim/z-ai/glm-5.1 | task/analysis | +0.3462 |
| nvidia_nim/z-ai/glm-5.1 | task/creative | +0.1856 |
| nvidia_nim/z-ai/glm-5.1 | task/generation | +0.0053 |
| nvidia_nim/z-ai/glm-5.1 | task/lookup | +0.0000 |
| nvidia_nim/z-ai/glm-5.1 | task/reasoning | +0.0000 |
| nvidia_nim/z-ai/glm-5.1 | task/refactoring | +0.5000 |
| nvidia_nim/z-ai/glm-5.1 | task/summarization | +0.5185 |
| nvidia_nim/z-ai/glm-5.1 | task/translation | +0.0385 |
