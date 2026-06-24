# Model Spectrography Report

- **Run ID:** 20260621-025704-e31b5ce0
- **Started:** 2026-06-21T02:57:04.442379+00:00
- **Completed:** 2026-06-21T17:40:57.488959+00:00
- **Judge model:** nvidia_nim/qwen/qwen3.5-397b-a17b
- **Models evaluated:** 34
- **Total probe evaluations:** 478
- **Errors:** 306
- **Self-evaluations:** 9

## Model Rankings (Overall Average)

| Rank | Model | Avg Score |
|------|-------|-----------|
| 1 | nvidia_nim/google/gemma-4-31b-it | 0.7540 |
| 2 | nvidia_nim/meta/llama-4-maverick-17b-128e-instruct | 0.7316 |
| 3 | nvidia_nim/stepfun-ai/step-3.5-flash | 0.6838 |
| 4 | nvidia_nim/mistralai/mistral-medium-3.5-128b | 0.6475 |
| 5 | nvidia_nim/nvidia/nemotron-3-super-120b-a12b | 0.5847 |
| 6 | nvidia_nim/qwen/qwen3-next-80b-a3b-instruct | 0.5830 |
| 7 | nvidia_nim/meta/llama-3.1-70b-instruct | 0.5639 |
| 8 | nvidia_nim/mistralai/mistral-small-4-119b-2603 | 0.5631 |
| 9 | nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512 | 0.5585 |
| 10 | nvidia_nim/qwen/qwen3.5-397b-a17b | 0.5560 |
| 11 | nvidia_nim/z-ai/glm-5.1 | 0.5465 |
| 12 | nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2 | 0.5453 |
| 13 | nvidia_nim/mistralai/mistral-nemotron | 0.5299 |
| 14 | nvidia_nim/openai/gpt-oss-20b | 0.5294 |
| 15 | nvidia_nim/mistralai/ministral-14b-instruct-2512 | 0.5280 |
| 16 | nvidia_nim/deepseek-ai/deepseek-v4-flash | 0.5098 |
| 17 | nvidia_nim/qwen/qwen3.5-122b-a10b | 0.5085 |
| 18 | nvidia_nim/deepseek-ai/deepseek-v4-pro | 0.5064 |
| 19 | nvidia_nim/minimaxai/minimax-m3 | 0.5014 |
| 20 | nvidia_nim/meta/llama-3.3-70b-instruct | 0.4899 |
| 21 | nvidia_nim/meta/llama-3.1-8b-instruct | 0.4819 |
| 22 | nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1 | 0.4774 |
| 23 | nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | 0.4599 |
| 24 | nvidia_nim/meta/llama-3.2-3b-instruct | 0.4492 |
| 25 | nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b | 0.4429 |
| 26 | nvidia_nim/openai/gpt-oss-120b | 0.4411 |
| 27 | nvidia_nim/minimaxai/minimax-m2.7 | 0.4302 |
| 28 | nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 | 0.4160 |
| 29 | nvidia_nim/google/gemma-3n-e4b-it | 0.4034 |
| 30 | nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1 | 0.3567 |
| 31 | nvidia_nim/moonshotai/kimi-k2.6 | 0.3420 |
| 32 | nvidia_nim/google/gemma-2-2b-it | 0.3313 |
| 33 | nvidia_nim/sarvamai/sarvam-m | 0.2883 |
| 34 | nvidia_nim/nvidia/nemotron-3-nano-30b-a3b | 0.2585 |

## Dimension Rankings

**domain/business:** nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/google/gemma-2-2b-it, nvidia_nim/google/gemma-3n-e4b-it
**domain/code:** nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/stepfun-ai/step-3.5-flash, nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2, nvidia_nim/nvidia/nemotron-3-super-120b-a12b
**domain/creative_writing:** nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/meta/llama-3.2-3b-instruct, nvidia_nim/meta/llama-3.1-8b-instruct, nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/google/gemma-4-31b-it
**domain/general:** nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/google/gemma-2-2b-it, nvidia_nim/google/gemma-3n-e4b-it
**domain/legal:** nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/google/gemma-2-2b-it, nvidia_nim/google/gemma-3n-e4b-it
**domain/technical:** nvidia_nim/qwen/qwen3.5-397b-a17b, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/stepfun-ai/step-3.5-flash, nvidia_nim/qwen/qwen3-next-80b-a3b-instruct
**qs/balanced:** nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/qwen/qwen3-next-80b-a3b-instruct, nvidia_nim/nvidia/nemotron-3-super-120b-a12b, nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1
**qs/quality:** nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/stepfun-ai/step-3.5-flash, nvidia_nim/mistralai/mistral-medium-3.5-128b, nvidia_nim/deepseek-ai/deepseek-v4-flash
**qs/speed:** nvidia_nim/qwen/qwen3.5-397b-a17b, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/stepfun-ai/step-3.5-flash, nvidia_nim/qwen/qwen3-next-80b-a3b-instruct, nvidia_nim/openai/gpt-oss-20b
**task/analysis:** nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/nvidia/nemotron-3-super-120b-a12b, nvidia_nim/mistralai/mistral-medium-3.5-128b, nvidia_nim/z-ai/glm-5.1
**task/creative:** nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/meta/llama-3.2-3b-instruct, nvidia_nim/meta/llama-3.1-8b-instruct, nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/google/gemma-4-31b-it
**task/generation:** nvidia_nim/qwen/qwen3.5-397b-a17b, nvidia_nim/meta/llama-4-maverick-17b-128e-instruct, nvidia_nim/meta/llama-3.1-70b-instruct, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/stepfun-ai/step-3.5-flash
**task/lookup:** nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/google/gemma-2-2b-it, nvidia_nim/google/gemma-3n-e4b-it
**task/reasoning:** nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/google/gemma-2-2b-it, nvidia_nim/google/gemma-3n-e4b-it
**task/refactoring:** nvidia_nim/minimaxai/minimax-m3, nvidia_nim/qwen/qwen3.5-122b-a10b, nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/deepseek-ai/deepseek-v4-pro
**task/summarization:** nvidia_nim/qwen/qwen3.5-397b-a17b, nvidia_nim/google/gemma-4-31b-it, nvidia_nim/stepfun-ai/step-3.5-flash, nvidia_nim/openai/gpt-oss-20b, nvidia_nim/openai/gpt-oss-120b
**task/translation:** nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct, nvidia_nim/deepseek-ai/deepseek-v4-flash, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/google/gemma-2-2b-it, nvidia_nim/google/gemma-3n-e4b-it

## Proficiencies & Deficiencies

**nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct**
- Top: qs/quality (0.63), task/analysis (0.62), task/summarization (0.50)
- Low: domain/technical (0.31), task/creative (0.27), domain/creative_writing (0.27)

**nvidia_nim/deepseek-ai/deepseek-v4-flash**
- Top: qs/quality (0.87), domain/code (0.81), task/analysis (0.73)
- Low: task/creative (0.31), domain/creative_writing (0.31), task/summarization (0.26)

**nvidia_nim/deepseek-ai/deepseek-v4-pro**
- Top: qs/quality (0.83), domain/code (0.72), task/analysis (0.69)
- Low: task/creative (0.35), domain/creative_writing (0.35), task/summarization (0.30)

**nvidia_nim/google/gemma-2-2b-it**
- Top: task/lookup (0.50), task/translation (0.50), task/reasoning (0.50)
- Low: task/generation (0.12), task/analysis (0.08), task/summarization (0.07)

**nvidia_nim/google/gemma-3n-e4b-it**
- Top: task/lookup (0.50), task/translation (0.50), task/reasoning (0.50)
- Low: qs/speed (0.28), domain/technical (0.22), task/summarization (0.11)

**nvidia_nim/google/gemma-4-31b-it**
- Top: domain/code (0.97), qs/quality (0.97), qs/speed (0.97)
- Low: domain/business (0.50), domain/general (0.50), domain/legal (0.50)

**nvidia_nim/meta/llama-3.1-70b-instruct**
- Top: task/generation (0.94), task/creative (0.88), domain/creative_writing (0.88)
- Low: qs/speed (0.31), domain/technical (0.25), task/summarization (0.15)

**nvidia_nim/meta/llama-3.1-8b-instruct**
- Top: task/creative (0.92), domain/creative_writing (0.92), domain/code (0.62)
- Low: domain/technical (0.19), task/summarization (0.19), task/generation (0.09)

**nvidia_nim/meta/llama-3.2-3b-instruct**
- Top: task/creative (0.96), domain/creative_writing (0.96), task/lookup (0.50)
- Low: task/analysis (0.23), task/summarization (0.22), domain/code (0.09)

**nvidia_nim/meta/llama-3.3-70b-instruct**
- Top: qs/balanced (0.61), task/analysis (0.54), task/summarization (0.50)
- Low: domain/code (0.44), domain/technical (0.41), qs/quality (0.40)

**nvidia_nim/meta/llama-4-maverick-17b-128e-instruct**
- Top: task/analysis (1.00), task/creative (1.00), domain/code (1.00)
- Low: domain/general (0.50), domain/legal (0.50), qs/speed (0.50)

**nvidia_nim/minimaxai/minimax-m2.7**
- Top: domain/technical (0.59), task/summarization (0.52), task/lookup (0.50)
- Low: qs/speed (0.14), task/generation (0.03), domain/code (0.03)

**nvidia_nim/minimaxai/minimax-m3**
- Top: task/refactoring (1.00), qs/quality (0.57), qs/balanced (0.57)
- Low: domain/technical (0.44), task/summarization (0.37), task/analysis (0.12)

**nvidia_nim/mistralai/ministral-14b-instruct-2512**
- Top: qs/quality (0.70), domain/technical (0.62), qs/speed (0.62)
- Low: domain/code (0.47), task/analysis (0.46), qs/balanced (0.39)

**nvidia_nim/mistralai/mistral-large-3-675b-instruct-2512**
- Top: task/generation (0.75), qs/balanced (0.74), domain/technical (0.66)
- Low: domain/legal (0.50), qs/quality (0.50), task/analysis (0.31)

**nvidia_nim/mistralai/mistral-medium-3.5-128b**
- Top: qs/quality (0.90), task/analysis (0.88), domain/code (0.84)
- Low: domain/business (0.50), domain/general (0.50), domain/legal (0.50)

**nvidia_nim/mistralai/mistral-nemotron**
- Top: task/analysis (0.77), domain/code (0.69), task/generation (0.66)
- Low: domain/technical (0.47), task/summarization (0.41), task/refactoring (0.25)

**nvidia_nim/mistralai/mistral-small-4-119b-2603**
- Top: task/analysis (0.81), qs/quality (0.80), domain/code (0.78)
- Low: domain/creative_writing (0.50), qs/balanced (0.50), task/summarization (0.44)

**nvidia_nim/moonshotai/kimi-k2.6**
- Top: task/creative (0.69), domain/creative_writing (0.69), task/summarization (0.67)
- Low: task/analysis (0.00), domain/code (0.00), qs/balanced (0.00)

**nvidia_nim/nvidia/llama-3.1-nemotron-nano-8b-v1**
- Top: qs/balanced (0.83), task/summarization (0.50), task/lookup (0.50)
- Low: task/analysis (0.04), qs/quality (0.03), qs/speed (0.00)

**nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1**
- Top: task/creative (0.73), domain/creative_writing (0.73), domain/technical (0.72)
- Low: task/generation (0.22), domain/code (0.22), qs/balanced (0.17)

**nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5**
- Top: task/lookup (0.50), task/generation (0.50), task/translation (0.50)
- Low: task/summarization (0.04), qs/speed (0.03), domain/technical (0.00)

**nvidia_nim/nvidia/nemotron-3-nano-30b-a3b**
- Top: task/lookup (0.50), task/translation (0.50), task/reasoning (0.50)
- Low: domain/creative_writing (0.04), domain/technical (0.03), task/summarization (0.00)

**nvidia_nim/nvidia/nemotron-3-super-120b-a12b**
- Top: task/analysis (0.92), domain/code (0.88), qs/balanced (0.87)
- Low: task/generation (0.34), qs/quality (0.30), domain/technical (0.12)

**nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b**
- Top: task/summarization (0.78), qs/speed (0.76), domain/technical (0.75)
- Low: qs/quality (0.17), task/creative (0.00), domain/creative_writing (0.00)

**nvidia_nim/nvidia/nvidia-nemotron-nano-9b-v2**
- Top: domain/code (0.91), task/summarization (0.81), task/generation (0.81)
- Low: task/creative (0.23), domain/creative_writing (0.23), qs/quality (0.20)

**nvidia_nim/openai/gpt-oss-120b**
- Top: task/summarization (0.85), qs/speed (0.83), domain/technical (0.81)
- Low: task/creative (0.12), domain/creative_writing (0.12), qs/balanced (0.04)

**nvidia_nim/openai/gpt-oss-20b**
- Top: task/summarization (0.89), qs/speed (0.86), domain/technical (0.84)
- Low: domain/code (0.41), task/generation (0.28), qs/balanced (0.22)

**nvidia_nim/qwen/qwen3-next-80b-a3b-instruct**
- Top: qs/balanced (0.91), qs/speed (0.90), domain/technical (0.88)
- Low: domain/creative_writing (0.50), task/analysis (0.42), qs/quality (0.37)

**nvidia_nim/qwen/qwen3.5-122b-a10b**
- Top: task/refactoring (0.75), qs/quality (0.73), domain/technical (0.53)
- Low: task/generation (0.41), domain/code (0.38), qs/balanced (0.35)

**nvidia_nim/qwen/qwen3.5-397b-a17b**
- Top: task/summarization (1.00), task/generation (1.00), domain/technical (1.00)
- Low: qs/quality (0.27), task/creative (0.08), domain/creative_writing (0.08)

**nvidia_nim/sarvamai/sarvam-m**
- Top: task/lookup (0.50), task/translation (0.50), task/reasoning (0.50)
- Low: domain/technical (0.06), domain/code (0.06), qs/quality (0.00)

**nvidia_nim/stepfun-ai/step-3.5-flash**
- Top: domain/code (0.94), qs/quality (0.93), qs/speed (0.93)
- Low: domain/general (0.50), domain/legal (0.50), qs/balanced (0.50)

**nvidia_nim/z-ai/glm-5.1**
- Top: task/analysis (0.85), task/generation (0.72), qs/balanced (0.70)
- Low: domain/legal (0.50), task/summarization (0.48), task/refactoring (0.00)

## Calibration Deltas

| Model | Dimension | Delta |
|-------|-----------|-------|
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/business | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/code | +0.1562 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/creative_writing | +0.2308 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/general | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/legal | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | domain/technical | +0.1875 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | qs/balanced | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | qs/quality | +0.1333 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | qs/speed | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/analysis | +0.1154 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/creative | +0.2308 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/generation | +0.1250 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/lookup | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/reasoning | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/refactoring | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/summarization | +0.0000 |
| nvidia_nim/abacusai/dracarys-llama-3.1-70b-instruct | task/translation | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/business | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/code | +0.8125 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/creative_writing | +0.1923 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/general | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/legal | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | domain/technical | +0.1562 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | qs/balanced | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | qs/quality | +0.8667 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | qs/speed | +0.0862 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/analysis | +0.2308 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/creative | +0.1923 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/generation | +0.6250 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/lookup | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/reasoning | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/refactoring | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/summarization | +0.2407 |
| nvidia_nim/deepseek-ai/deepseek-v4-flash | task/translation | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/business | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/code | +0.4688 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/creative_writing | +0.1538 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/general | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/legal | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/technical | +0.1250 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | qs/balanced | +0.0217 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | qs/quality | +0.5833 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | qs/speed | +0.0517 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/analysis | +0.1923 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/creative | +0.1538 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/generation | +0.2812 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/lookup | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/reasoning | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/refactoring | +0.0000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/summarization | +0.2037 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/translation | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/business | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/code | +0.4688 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/creative_writing | +0.0769 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/general | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/legal | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | domain/technical | +0.2812 |
| nvidia_nim/google/gemma-3n-e4b-it | qs/balanced | +0.1957 |
| nvidia_nim/google/gemma-3n-e4b-it | qs/quality | +0.2833 |
| nvidia_nim/google/gemma-3n-e4b-it | qs/speed | +0.2241 |
| nvidia_nim/google/gemma-3n-e4b-it | task/analysis | +0.1154 |
| nvidia_nim/google/gemma-3n-e4b-it | task/creative | +0.0769 |
| nvidia_nim/google/gemma-3n-e4b-it | task/generation | +0.2812 |
| nvidia_nim/google/gemma-3n-e4b-it | task/lookup | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | task/reasoning | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | task/refactoring | +0.0000 |
| nvidia_nim/google/gemma-3n-e4b-it | task/summarization | +0.3889 |
| nvidia_nim/google/gemma-3n-e4b-it | task/translation | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | domain/business | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | domain/code | +0.0312 |
| nvidia_nim/google/gemma-4-31b-it | domain/creative_writing | +0.3462 |
| nvidia_nim/google/gemma-4-31b-it | domain/general | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | domain/legal | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | domain/technical | +0.4375 |
| nvidia_nim/google/gemma-4-31b-it | qs/balanced | +0.4565 |
| nvidia_nim/google/gemma-4-31b-it | qs/quality | +0.0333 |
| nvidia_nim/google/gemma-4-31b-it | qs/speed | +0.4655 |
| nvidia_nim/google/gemma-4-31b-it | task/analysis | +0.4615 |
| nvidia_nim/google/gemma-4-31b-it | task/creative | +0.3462 |
| nvidia_nim/google/gemma-4-31b-it | task/generation | +0.0938 |
| nvidia_nim/google/gemma-4-31b-it | task/lookup | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | task/reasoning | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | task/refactoring | +0.0000 |
| nvidia_nim/google/gemma-4-31b-it | task/summarization | +0.4630 |
| nvidia_nim/google/gemma-4-31b-it | task/translation | +0.0000 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/business | +0.5000 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/code | +0.2500 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/creative_writing | +0.6923 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/general | +0.2500 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/legal | +0.5000 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/technical | +0.1562 |
| nvidia_nim/moonshotai/kimi-k2.6 | qs/balanced | +0.2500 |
| nvidia_nim/moonshotai/kimi-k2.6 | qs/quality | +0.1500 |
| nvidia_nim/moonshotai/kimi-k2.6 | qs/speed | +0.0690 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/analysis | +0.2500 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/creative | +0.4423 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/generation | +0.2500 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/lookup | +0.5000 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/reasoning | +0.5000 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/refactoring | +0.0000 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/summarization | +0.6667 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/translation | +0.5000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/business | +0.5000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/code | +0.2188 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/creative_writing | +0.2564 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/general | +0.2500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/legal | +0.1667 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/technical | +0.2500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | qs/balanced | +0.2500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | qs/quality | +0.2333 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | qs/speed | +0.5000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/analysis | +0.2500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/creative | +0.6731 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/generation | +0.0000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/lookup | +0.5000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/reasoning | +0.1667 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/refactoring | +0.2500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/summarization | +0.3333 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/translation | +0.1667 |
