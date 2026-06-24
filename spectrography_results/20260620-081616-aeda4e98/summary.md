# Model Spectrography Report

- **Run ID:** 20260620-081616-aeda4e98
- **Started:** 2026-06-20T08:16:16.444050+00:00
- **Completed:** 2026-06-21T02:09:55.374426+00:00
- **Judge model:** nvidia_nim/qwen/qwen3.5-397b-a17b
- **Models evaluated:** 5
- **Total probe evaluations:** 735
- **Errors:** 551
- **Self-evaluations:** 73

## Model Rankings (Overall Average)

| Rank | Model | Avg Score |
|------|-------|-----------|
| 1 | nvidia_nim/deepseek-ai/deepseek-v4-pro | 0.8431 |
| 2 | nvidia_nim/qwen/qwen3.5-397b-a17b | 0.7010 |
| 3 | groq/llama-3.3-70b-versatile | 0.5294 |
| 4 | gemini/gemini-2.5-flash | 0.2794 |
| 5 | nvidia_nim/moonshotai/kimi-k2.6 | 0.1471 |

## Dimension Rankings

**domain/business:** nvidia_nim/qwen/qwen3.5-397b-a17b, gemini/gemini-2.5-flash, groq/llama-3.3-70b-versatile, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/moonshotai/kimi-k2.6
**domain/code:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6, gemini/gemini-2.5-flash
**domain/creative_writing:** nvidia_nim/deepseek-ai/deepseek-v4-pro, groq/llama-3.3-70b-versatile, gemini/gemini-2.5-flash, nvidia_nim/qwen/qwen3.5-397b-a17b, nvidia_nim/moonshotai/kimi-k2.6
**domain/general:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6, gemini/gemini-2.5-flash
**domain/legal:** groq/llama-3.3-70b-versatile, nvidia_nim/qwen/qwen3.5-397b-a17b, gemini/gemini-2.5-flash, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/moonshotai/kimi-k2.6
**domain/technical:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6, gemini/gemini-2.5-flash
**qs/balanced:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6, gemini/gemini-2.5-flash
**qs/quality:** nvidia_nim/deepseek-ai/deepseek-v4-pro, groq/llama-3.3-70b-versatile, nvidia_nim/qwen/qwen3.5-397b-a17b, nvidia_nim/moonshotai/kimi-k2.6, gemini/gemini-2.5-flash
**qs/speed:** nvidia_nim/deepseek-ai/deepseek-v4-pro, gemini/gemini-2.5-flash, nvidia_nim/qwen/qwen3.5-397b-a17b, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6
**task/analysis:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6, gemini/gemini-2.5-flash
**task/creative:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6, gemini/gemini-2.5-flash
**task/generation:** nvidia_nim/qwen/qwen3.5-397b-a17b, groq/llama-3.3-70b-versatile, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/moonshotai/kimi-k2.6, gemini/gemini-2.5-flash
**task/lookup:** nvidia_nim/qwen/qwen3.5-397b-a17b, gemini/gemini-2.5-flash, groq/llama-3.3-70b-versatile, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/moonshotai/kimi-k2.6
**task/reasoning:** gemini/gemini-2.5-flash, groq/llama-3.3-70b-versatile, nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, nvidia_nim/moonshotai/kimi-k2.6
**task/refactoring:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, nvidia_nim/moonshotai/kimi-k2.6, groq/llama-3.3-70b-versatile, gemini/gemini-2.5-flash
**task/summarization:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, gemini/gemini-2.5-flash, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6
**task/translation:** nvidia_nim/deepseek-ai/deepseek-v4-pro, nvidia_nim/qwen/qwen3.5-397b-a17b, gemini/gemini-2.5-flash, groq/llama-3.3-70b-versatile, nvidia_nim/moonshotai/kimi-k2.6

## Proficiencies & Deficiencies

**gemini/gemini-2.5-flash**
- Top: task/reasoning (1.00), qs/speed (0.75), task/summarization (0.50)
- Low: domain/code (0.00), qs/quality (0.00), qs/balanced (0.00)

**groq/llama-3.3-70b-versatile**
- Top: domain/legal (1.00), task/generation (0.75), qs/quality (0.75)
- Low: task/translation (0.33), task/refactoring (0.25), qs/speed (0.25)

**nvidia_nim/deepseek-ai/deepseek-v4-pro**
- Top: task/analysis (1.00), task/summarization (1.00), task/creative (1.00)
- Low: task/reasoning (0.50), domain/business (0.50), domain/legal (0.33)

**nvidia_nim/moonshotai/kimi-k2.6**
- Top: task/refactoring (0.50), task/analysis (0.25), task/creative (0.25)
- Low: domain/legal (0.00), domain/business (0.00), qs/speed (0.00)

**nvidia_nim/qwen/qwen3.5-397b-a17b**
- Top: task/generation (1.00), task/lookup (1.00), domain/business (1.00)
- Low: qs/speed (0.50), task/reasoning (0.33), domain/creative_writing (0.33)

## Calibration Deltas

| Model | Dimension | Delta |
|-------|-----------|-------|
| gemini/gemini-2.5-flash | domain/business | +0.1500 |
| gemini/gemini-2.5-flash | domain/code | +0.7000 |
| gemini/gemini-2.5-flash | domain/creative_writing | +0.0000 |
| gemini/gemini-2.5-flash | domain/general | +0.8000 |
| gemini/gemini-2.5-flash | domain/legal | +0.0000 |
| gemini/gemini-2.5-flash | domain/technical | +0.7500 |
| gemini/gemini-2.5-flash | qs/balanced | +0.8000 |
| gemini/gemini-2.5-flash | qs/quality | +0.5000 |
| gemini/gemini-2.5-flash | qs/speed | +0.1500 |
| gemini/gemini-2.5-flash | task/analysis | +0.7500 |
| gemini/gemini-2.5-flash | task/creative | +0.6000 |
| gemini/gemini-2.5-flash | task/generation | +0.7000 |
| gemini/gemini-2.5-flash | task/lookup | +0.3500 |
| gemini/gemini-2.5-flash | task/reasoning | +0.3500 |
| gemini/gemini-2.5-flash | task/refactoring | +0.6500 |
| gemini/gemini-2.5-flash | task/summarization | +0.3000 |
| gemini/gemini-2.5-flash | task/translation | +0.2000 |
| groq/llama-3.3-70b-versatile | domain/business | +0.1500 |
| groq/llama-3.3-70b-versatile | domain/code | +0.2000 |
| groq/llama-3.3-70b-versatile | domain/creative_writing | +0.1667 |
| groq/llama-3.3-70b-versatile | domain/general | +0.2500 |
| groq/llama-3.3-70b-versatile | domain/legal | +0.5000 |
| groq/llama-3.3-70b-versatile | domain/technical | +0.3000 |
| groq/llama-3.3-70b-versatile | qs/balanced | +0.2000 |
| groq/llama-3.3-70b-versatile | qs/quality | +0.3000 |
| groq/llama-3.3-70b-versatile | qs/speed | +0.7000 |
| groq/llama-3.3-70b-versatile | task/analysis | +0.3000 |
| groq/llama-3.3-70b-versatile | task/creative | +0.0500 |
| groq/llama-3.3-70b-versatile | task/generation | +0.0500 |
| groq/llama-3.3-70b-versatile | task/lookup | +0.3000 |
| groq/llama-3.3-70b-versatile | task/reasoning | +0.0333 |
| groq/llama-3.3-70b-versatile | task/refactoring | +0.3500 |
| groq/llama-3.3-70b-versatile | task/summarization | +0.4167 |
| groq/llama-3.3-70b-versatile | task/translation | +0.3167 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/business | +0.2000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/code | +0.1000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/creative_writing | +0.5000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/general | +0.2500 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/legal | +0.3167 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | domain/technical | +0.1000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | qs/balanced | +0.2500 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | qs/quality | +0.1000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | qs/speed | +0.5500 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/analysis | +0.1000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/creative | +0.3500 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/generation | +0.3000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/lookup | +0.1500 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/reasoning | +0.4500 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/refactoring | +0.1500 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/summarization | +0.2000 |
| nvidia_nim/deepseek-ai/deepseek-v4-pro | task/translation | +0.3000 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/business | +0.5000 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/code | +0.6500 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/creative_writing | +0.5000 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/general | +0.4000 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/legal | +0.5000 |
| nvidia_nim/moonshotai/kimi-k2.6 | domain/technical | +0.6000 |
| nvidia_nim/moonshotai/kimi-k2.6 | qs/balanced | +0.5500 |
| nvidia_nim/moonshotai/kimi-k2.6 | qs/quality | +0.6000 |
| nvidia_nim/moonshotai/kimi-k2.6 | qs/speed | +0.5500 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/analysis | +0.5500 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/creative | +0.4000 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/generation | +0.6000 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/lookup | +0.6000 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/reasoning | +0.8000 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/refactoring | +0.3500 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/summarization | +0.7000 |
| nvidia_nim/moonshotai/kimi-k2.6 | task/translation | +0.6000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/business | +0.2500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/code | +0.1000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/creative_writing | +0.3167 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/general | +0.0500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/legal | +0.0333 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | domain/technical | +0.1500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | qs/balanced | +0.0500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | qs/quality | +0.4500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | qs/speed | +0.1500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/analysis | +0.1500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/creative | +0.0500 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/generation | +0.2000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/lookup | +0.3000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/reasoning | +0.6167 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/refactoring | +0.0000 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/summarization | +0.1833 |
| nvidia_nim/qwen/qwen3.5-397b-a17b | task/translation | +0.0833 |
