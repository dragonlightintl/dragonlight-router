# Model Spectrography Report

- **Run ID:** 20260621-025642-363db5d4
- **Started:** 2026-06-21T02:56:42.605557+00:00
- **Completed:** 2026-06-21T03:46:29.889788+00:00
- **Judge model:** nvidia_nim/qwen/qwen3.5-397b-a17b
- **Models evaluated:** 8
- **Total probe evaluations:** 648
- **Errors:** 562
- **Self-evaluations:** 0

## Model Rankings (Overall Average)

| Rank | Model | Avg Score |
|------|-------|-----------|
| 1 | groq/llama-3.3-70b-versatile | 0.6667 |
| 2 | groq/meta-llama/llama-4-scout-17b-16e-instruct | 0.6619 |
| 3 | groq/openai/gpt-oss-120b | 0.6394 |
| 4 | groq/openai/gpt-oss-20b | 0.5490 |
| 5 | groq/qwen/qwen3-32b | 0.4594 |
| 6 | groq/llama-3.1-8b-instant | 0.4254 |
| 7 | groq/allam-2-7b | 0.3227 |
| 8 | groq/qwen/qwen3.6-27b | 0.2756 |

## Dimension Rankings

**domain/business:** groq/openai/gpt-oss-120b, groq/llama-3.3-70b-versatile, groq/allam-2-7b, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-20b
**domain/code:** groq/openai/gpt-oss-20b, groq/openai/gpt-oss-120b, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/llama-3.1-8b-instant
**domain/creative_writing:** groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b, groq/allam-2-7b, groq/llama-3.3-70b-versatile, groq/openai/gpt-oss-20b
**domain/general:** groq/qwen/qwen3-32b, groq/llama-3.3-70b-versatile, groq/llama-3.1-8b-instant, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-20b
**domain/legal:** groq/allam-2-7b, groq/openai/gpt-oss-20b, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**domain/technical:** groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b, groq/llama-3.1-8b-instant, groq/qwen/qwen3-32b
**qs/balanced:** groq/llama-3.3-70b-versatile, groq/llama-3.1-8b-instant, groq/qwen/qwen3-32b, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**qs/quality:** groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b, groq/llama-3.3-70b-versatile, groq/openai/gpt-oss-20b, groq/llama-3.1-8b-instant
**qs/speed:** groq/openai/gpt-oss-20b, groq/openai/gpt-oss-120b, groq/qwen/qwen3-32b, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/llama-3.1-8b-instant
**task/analysis:** groq/openai/gpt-oss-120b, groq/llama-3.3-70b-versatile, groq/qwen/qwen3-32b, groq/llama-3.1-8b-instant, groq/openai/gpt-oss-20b
**task/creative:** groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/allam-2-7b, groq/openai/gpt-oss-20b, groq/qwen/qwen3-32b
**task/generation:** groq/openai/gpt-oss-20b, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b, groq/llama-3.1-8b-instant, groq/llama-3.3-70b-versatile
**task/lookup:** groq/qwen/qwen3-32b, groq/allam-2-7b, groq/llama-3.1-8b-instant, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b
**task/reasoning:** groq/qwen/qwen3.6-27b, groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b, groq/qwen/qwen3-32b
**task/refactoring:** groq/llama-3.3-70b-versatile, groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/llama-3.1-8b-instant, groq/openai/gpt-oss-120b, groq/openai/gpt-oss-20b
**task/summarization:** groq/llama-3.1-8b-instant, groq/qwen/qwen3-32b, groq/allam-2-7b, groq/llama-3.3-70b-versatile, groq/openai/gpt-oss-120b
**task/translation:** groq/meta-llama/llama-4-scout-17b-16e-instruct, groq/openai/gpt-oss-120b, groq/llama-3.3-70b-versatile, groq/openai/gpt-oss-20b, groq/qwen/qwen3-32b

## Proficiencies & Deficiencies

**groq/allam-2-7b**
- Top: domain/legal (1.00), task/lookup (0.50), task/summarization (0.50)
- Low: qs/speed (0.14), task/translation (0.00), domain/general (0.00)

**groq/llama-3.1-8b-instant**
- Top: task/summarization (1.00), qs/balanced (0.86), task/refactoring (0.60)
- Low: task/reasoning (0.00), task/creative (0.00), domain/creative_writing (0.00)

**groq/llama-3.3-70b-versatile**
- Top: task/refactoring (1.00), task/creative (1.00), domain/technical (1.00)
- Low: task/generation (0.43), qs/speed (0.29), task/lookup (0.00)

**groq/meta-llama/llama-4-scout-17b-16e-instruct**
- Top: task/translation (1.00), domain/creative_writing (1.00), qs/quality (1.00)
- Low: domain/legal (0.50), task/analysis (0.33), task/summarization (0.33)

**groq/openai/gpt-oss-120b**
- Top: task/analysis (1.00), domain/business (1.00), domain/code (0.86)
- Low: qs/balanced (0.43), task/creative (0.33), domain/general (0.25)

**groq/openai/gpt-oss-20b**
- Top: task/generation (1.00), domain/code (1.00), qs/speed (1.00)
- Low: task/reasoning (0.29), domain/technical (0.17), qs/balanced (0.14)

**groq/qwen/qwen3-32b**
- Top: task/lookup (1.00), domain/general (1.00), qs/speed (0.71)
- Low: task/refactoring (0.00), domain/code (0.00), domain/business (0.00)

**groq/qwen/qwen3.6-27b**
- Top: task/reasoning (1.00), task/lookup (0.50), task/translation (0.50)
- Low: qs/speed (0.00), qs/quality (0.00), qs/balanced (0.00)

## Calibration Deltas

| Model | Dimension | Delta |
|-------|-----------|-------|
| groq/allam-2-7b | domain/business | +0.0000 |
| groq/allam-2-7b | domain/code | +0.4285 |
| groq/allam-2-7b | domain/creative_writing | +0.0000 |
| groq/allam-2-7b | domain/general | +0.5000 |
| groq/allam-2-7b | domain/legal | +0.5000 |
| groq/allam-2-7b | domain/technical | +0.1667 |
| groq/allam-2-7b | qs/balanced | +0.2143 |
| groq/allam-2-7b | qs/quality | +0.3810 |
| groq/allam-2-7b | qs/speed | +0.0238 |
| groq/allam-2-7b | task/analysis | +0.3333 |
| groq/allam-2-7b | task/creative | +0.0000 |
| groq/allam-2-7b | task/generation | +0.0000 |
| groq/allam-2-7b | task/lookup | +0.0000 |
| groq/allam-2-7b | task/reasoning | +0.3571 |
| groq/allam-2-7b | task/refactoring | +0.6333 |
| groq/allam-2-7b | task/summarization | +0.0000 |
| groq/allam-2-7b | task/translation | +0.5000 |
| groq/llama-3.1-8b-instant | domain/business | +0.2500 |
| groq/llama-3.1-8b-instant | domain/code | +0.0000 |
| groq/llama-3.1-8b-instant | domain/creative_writing | +0.5000 |
| groq/llama-3.1-8b-instant | domain/general | +0.0000 |
| groq/llama-3.1-8b-instant | domain/legal | +0.1667 |
| groq/llama-3.1-8b-instant | domain/technical | +0.0000 |
| groq/llama-3.1-8b-instant | qs/balanced | +0.1429 |
| groq/llama-3.1-8b-instant | qs/quality | +0.0953 |
| groq/llama-3.1-8b-instant | qs/speed | +0.0953 |
| groq/llama-3.1-8b-instant | task/analysis | +0.0000 |
| groq/llama-3.1-8b-instant | task/creative | +0.5000 |
| groq/llama-3.1-8b-instant | task/generation | +0.1429 |
| groq/llama-3.1-8b-instant | task/lookup | +0.0000 |
| groq/llama-3.1-8b-instant | task/reasoning | +0.5000 |
| groq/llama-3.1-8b-instant | task/refactoring | +0.1000 |
| groq/llama-3.1-8b-instant | task/summarization | +0.5000 |
| groq/llama-3.1-8b-instant | task/translation | +0.1667 |
| groq/llama-3.3-70b-versatile | domain/business | +0.2500 |
| groq/llama-3.3-70b-versatile | domain/code | +0.0000 |
| groq/llama-3.3-70b-versatile | domain/creative_writing | +0.0000 |
| groq/llama-3.3-70b-versatile | domain/general | +0.2500 |
| groq/llama-3.3-70b-versatile | domain/legal | +0.0000 |
| groq/llama-3.3-70b-versatile | domain/technical | +0.5000 |
| groq/llama-3.3-70b-versatile | qs/balanced | +0.3333 |
| groq/llama-3.3-70b-versatile | qs/quality | +0.1190 |
| groq/llama-3.3-70b-versatile | qs/speed | +0.2143 |
| groq/llama-3.3-70b-versatile | task/analysis | +0.3333 |
| groq/llama-3.3-70b-versatile | task/creative | +0.5000 |
| groq/llama-3.3-70b-versatile | task/generation | +0.0000 |
| groq/llama-3.3-70b-versatile | task/lookup | +0.5000 |
| groq/llama-3.3-70b-versatile | task/reasoning | +0.3571 |
| groq/llama-3.3-70b-versatile | task/refactoring | +0.0000 |
| groq/llama-3.3-70b-versatile | task/summarization | +0.0000 |
| groq/llama-3.3-70b-versatile | task/translation | +0.0000 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | domain/business | +0.0000 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | domain/code | +0.2857 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | domain/creative_writing | +0.5000 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | domain/general | +0.0000 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | domain/legal | +0.0000 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | domain/technical | +0.3333 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | qs/balanced | +0.2619 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | qs/quality | +0.8333 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | qs/speed | +0.0953 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | task/analysis | +0.1667 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | task/creative | +0.1667 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | task/generation | +0.2857 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | task/lookup | +0.0000 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | task/reasoning | +0.2143 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | task/refactoring | +0.4667 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | task/summarization | +0.1667 |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | task/translation | +0.5000 |
| groq/openai/gpt-oss-120b | domain/business | +0.5000 |
| groq/openai/gpt-oss-120b | domain/code | +0.0000 |
| groq/openai/gpt-oss-120b | domain/creative_writing | +0.1667 |
| groq/openai/gpt-oss-120b | domain/general | +0.2500 |
| groq/openai/gpt-oss-120b | domain/legal | +0.0000 |
| groq/openai/gpt-oss-120b | domain/technical | +0.1667 |
| groq/openai/gpt-oss-120b | qs/balanced | +0.0953 |
| groq/openai/gpt-oss-120b | qs/quality | +0.1429 |
| groq/openai/gpt-oss-120b | qs/speed | +0.0238 |
| groq/openai/gpt-oss-120b | task/analysis | +0.5000 |
| groq/openai/gpt-oss-120b | task/creative | +0.1667 |
| groq/openai/gpt-oss-120b | task/generation | +0.1428 |
| groq/openai/gpt-oss-120b | task/lookup | +0.0000 |
| groq/openai/gpt-oss-120b | task/reasoning | +0.0714 |
| groq/openai/gpt-oss-120b | task/refactoring | +0.1667 |
| groq/openai/gpt-oss-120b | task/summarization | +0.0000 |
| groq/openai/gpt-oss-120b | task/translation | +0.1667 |
| groq/openai/gpt-oss-20b | domain/business | +0.0000 |
| groq/openai/gpt-oss-20b | domain/code | +0.0000 |
| groq/openai/gpt-oss-20b | domain/creative_writing | +0.0000 |
| groq/openai/gpt-oss-20b | domain/general | +0.0000 |
| groq/openai/gpt-oss-20b | domain/legal | +0.1667 |
| groq/openai/gpt-oss-20b | domain/technical | +0.3333 |
| groq/openai/gpt-oss-20b | qs/balanced | +0.3571 |
| groq/openai/gpt-oss-20b | qs/quality | +0.0714 |
| groq/openai/gpt-oss-20b | qs/speed | +0.0000 |
| groq/openai/gpt-oss-20b | task/analysis | +0.0000 |
| groq/openai/gpt-oss-20b | task/creative | +0.0000 |
| groq/openai/gpt-oss-20b | task/generation | +0.0000 |
| groq/openai/gpt-oss-20b | task/lookup | +0.0000 |
| groq/openai/gpt-oss-20b | task/reasoning | +0.2143 |
| groq/openai/gpt-oss-20b | task/refactoring | +0.0000 |
| groq/openai/gpt-oss-20b | task/summarization | +0.0000 |
| groq/openai/gpt-oss-20b | task/translation | +0.0000 |
| groq/qwen/qwen3-32b | domain/business | +0.5000 |
| groq/qwen/qwen3-32b | domain/code | +0.1429 |
| groq/qwen/qwen3-32b | domain/creative_writing | +0.1667 |
| groq/qwen/qwen3-32b | domain/general | +0.5000 |
| groq/qwen/qwen3-32b | domain/legal | +0.0000 |
| groq/qwen/qwen3-32b | domain/technical | +0.0000 |
| groq/qwen/qwen3-32b | qs/balanced | +0.5476 |
| groq/qwen/qwen3-32b | qs/quality | +0.3571 |
| groq/qwen/qwen3-32b | qs/speed | +0.7143 |
| groq/qwen/qwen3-32b | task/analysis | +0.1667 |
| groq/qwen/qwen3-32b | task/creative | +0.0000 |
| groq/qwen/qwen3-32b | task/generation | +0.0000 |
| groq/qwen/qwen3-32b | task/lookup | +0.5000 |
| groq/qwen/qwen3-32b | task/reasoning | +0.0714 |
| groq/qwen/qwen3-32b | task/refactoring | +0.0000 |
| groq/qwen/qwen3-32b | task/summarization | +0.1667 |
| groq/qwen/qwen3-32b | task/translation | +0.0000 |
| groq/qwen/qwen3.6-27b | domain/business | +0.0000 |
| groq/qwen/qwen3.6-27b | domain/code | +0.2857 |
| groq/qwen/qwen3.6-27b | domain/creative_writing | +0.0000 |
| groq/qwen/qwen3.6-27b | domain/general | +0.0000 |
| groq/qwen/qwen3.6-27b | domain/legal | +0.5000 |
| groq/qwen/qwen3.6-27b | domain/technical | +0.5000 |
| groq/qwen/qwen3.6-27b | qs/balanced | +0.0000 |
| groq/qwen/qwen3.6-27b | qs/quality | +0.0000 |
| groq/qwen/qwen3.6-27b | qs/speed | +0.5000 |
| groq/qwen/qwen3.6-27b | task/analysis | +0.5000 |
| groq/qwen/qwen3.6-27b | task/creative | +0.0000 |
| groq/qwen/qwen3.6-27b | task/generation | +0.0000 |
| groq/qwen/qwen3.6-27b | task/lookup | +0.0000 |
| groq/qwen/qwen3.6-27b | task/reasoning | +0.5000 |
| groq/qwen/qwen3.6-27b | task/refactoring | +0.2333 |
| groq/qwen/qwen3.6-27b | task/summarization | +0.5000 |
| groq/qwen/qwen3.6-27b | task/translation | +0.0000 |
