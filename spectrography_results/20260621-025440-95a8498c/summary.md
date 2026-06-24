# Model Spectrography Report

- **Run ID:** 20260621-025440-95a8498c
- **Started:** 2026-06-21T02:54:40.003496+00:00
- **Completed:** 2026-06-21T02:56:34.547402+00:00
- **Judge model:** nvidia_nim/qwen/qwen3.5-397b-a17b
- **Models evaluated:** 8
- **Total probe evaluations:** 31
- **Errors:** 4
- **Self-evaluations:** 0

## Model Rankings (Overall Average)

| Rank | Model | Avg Score |
|------|-------|-----------|
| 1 | groq/openai/gpt-oss-120b | 0.5910 |
| 2 | groq/openai/gpt-oss-20b | 0.5882 |
| 3 | groq/llama-3.3-70b-versatile | 0.5672 |
| 4 | groq/llama-3.1-8b-instant | 0.5182 |
| 5 | groq/allam-2-7b | 0.5014 |
| 6 | groq/meta-llama/llama-4-scout-17b-16e-instruct | 0.4916 |
| 7 | groq/qwen/qwen3-32b | 0.3796 |
| 8 | groq/qwen/qwen3.6-27b | 0.3627 |

## Dimension Rankings

**domain/business:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**domain/code:** groq/openai/gpt-oss-20b, groq/openai/gpt-oss-120b, groq/llama-3.3-70b-versatile, groq/allam-2-7b, groq/llama-3.1-8b-instant
**domain/creative_writing:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**domain/general:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**domain/legal:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**domain/technical:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**qs/balanced:** groq/llama-3.1-8b-instant, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/llama-3.3-70b-versatile, groq/allam-2-7b, groq/openai/gpt-oss-20b
**qs/quality:** groq/openai/gpt-oss-120b, groq/llama-3.3-70b-versatile, groq/allam-2-7b, groq/openai/gpt-oss-20b, groq/qwen/qwen3-32b
**qs/speed:** groq/openai/gpt-oss-20b, groq/openai/gpt-oss-120b, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/llama-3.3-70b-versatile, groq/qwen/qwen3.6-27b
**task/analysis:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**task/creative:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**task/generation:** groq/openai/gpt-oss-20b, groq/openai/gpt-oss-120b, groq/llama-3.1-8b-instant, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/llama-3.3-70b-versatile
**task/lookup:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**task/reasoning:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**task/refactoring:** groq/llama-3.3-70b-versatile, groq/allam-2-7b, groq/openai/gpt-oss-120b, groq/llama-3.1-8b-instant, groq/openai/gpt-oss-20b
**task/summarization:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**task/translation:** groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b

## Proficiencies & Deficiencies

**groq/allam-2-7b**
- Top: task/refactoring (0.83), qs/quality (0.67), domain/code (0.57)
- Low: qs/balanced (0.50), task/generation (0.29), qs/speed (0.17)

**groq/llama-3.1-8b-instant**
- Top: qs/balanced (1.00), task/generation (0.71), task/refactoring (0.50)
- Low: domain/code (0.43), qs/quality (0.33), qs/speed (0.33)

**groq/llama-3.3-70b-versatile**
- Top: task/refactoring (1.00), qs/quality (0.83), domain/code (0.71)
- Low: domain/creative_writing (0.50), qs/speed (0.50), task/generation (0.43)

**groq/meta-llama/llama-4-scout-17b-16e-instruct**
- Top: qs/balanced (0.83), qs/speed (0.67), task/generation (0.57)
- Low: task/refactoring (0.33), domain/code (0.29), qs/quality (0.17)

**groq/openai/gpt-oss-120b**
- Top: qs/quality (1.00), task/generation (0.86), domain/code (0.86)
- Low: domain/technical (0.50), domain/creative_writing (0.50), qs/balanced (0.33)

**groq/openai/gpt-oss-20b**
- Top: task/generation (1.00), domain/code (1.00), qs/speed (1.00)
- Low: domain/creative_writing (0.50), qs/balanced (0.50), qs/quality (0.50)

**groq/qwen/qwen3-32b**
- Top: task/reasoning (0.50), task/translation (0.50), task/summarization (0.50)
- Low: domain/code (0.14), task/refactoring (0.00), qs/speed (0.00)

**groq/qwen/qwen3.6-27b**
- Top: task/reasoning (0.50), task/translation (0.50), task/summarization (0.50)
- Low: domain/code (0.00), qs/balanced (0.00), qs/quality (0.00)

## Calibration Deltas

| Model | Dimension | Delta |
|-------|-----------|-------|
| groq/llama-3.3-70b-versatile | domain/business | +0.0000 |
| groq/llama-3.3-70b-versatile | domain/code | +0.2143 |
| groq/llama-3.3-70b-versatile | domain/creative_writing | +0.1667 |
| groq/llama-3.3-70b-versatile | domain/general | +0.0000 |
| groq/llama-3.3-70b-versatile | domain/legal | +0.5000 |
| groq/llama-3.3-70b-versatile | domain/technical | +0.0000 |
| groq/llama-3.3-70b-versatile | qs/balanced | +0.1667 |
| groq/llama-3.3-70b-versatile | qs/quality | +0.0833 |
| groq/llama-3.3-70b-versatile | qs/speed | +0.2500 |
| groq/llama-3.3-70b-versatile | task/analysis | +0.0000 |
| groq/llama-3.3-70b-versatile | task/creative | +0.0000 |
| groq/llama-3.3-70b-versatile | task/generation | +0.3214 |
| groq/llama-3.3-70b-versatile | task/lookup | +0.0000 |
| groq/llama-3.3-70b-versatile | task/reasoning | +0.1667 |
| groq/llama-3.3-70b-versatile | task/refactoring | +0.7500 |
| groq/llama-3.3-70b-versatile | task/summarization | +0.1667 |
| groq/llama-3.3-70b-versatile | task/translation | +0.1667 |
