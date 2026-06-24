# Model Spectrography Report

- **Run ID:** 20260622-201018-fc2b54a8
- **Started:** 2026-06-22T20:10:18.801736+00:00
- **Completed:** 2026-06-22T22:32:44.218599+00:00
- **Judge model:** nvidia_nim/qwen/qwen3.5-397b-a17b
- **Models evaluated:** 6
- **Total probe evaluations:** 579
- **Errors:** 468
- **Self-evaluations:** 0

## Model Rankings (Overall Average)

| Rank | Model | Avg Score |
|------|-------|-----------|
| 1 | gemini/gemma-4-31b-it | 0.6931 |
| 2 | gemini/gemini-2.5-flash-lite | 0.6245 |
| 3 | gemini/gemma-4-26b-a4b-it | 0.6216 |
| 4 | nvidia_nim/microsoft/phi-4-mini-instruct | 0.5451 |
| 5 | gemini/gemini-3.5-flash | 0.3216 |
| 6 | gemini/gemini-3-flash-preview | 0.1941 |

## Dimension Rankings

**domain/business:** gemini/gemma-4-31b-it, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemini-3-flash-preview, gemini/gemini-3.5-flash, gemini/gemma-4-26b-a4b-it
**domain/code:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-31b-it, gemini/gemma-4-26b-a4b-it, gemini/gemini-3.5-flash, nvidia_nim/microsoft/phi-4-mini-instruct
**domain/creative_writing:** gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, gemini/gemini-2.5-flash-lite, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemini-3.5-flash
**domain/general:** gemini/gemini-2.5-flash-lite, gemini/gemini-3-flash-preview, gemini/gemini-3.5-flash, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it
**domain/legal:** gemini/gemma-4-31b-it, gemini/gemini-2.5-flash-lite, gemini/gemini-3-flash-preview, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemma-4-26b-a4b-it
**domain/technical:** nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemini-2.5-flash-lite, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, gemini/gemini-3-flash-preview
**qs/balanced:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-31b-it, gemini/gemma-4-26b-a4b-it, gemini/gemini-3-flash-preview, nvidia_nim/microsoft/phi-4-mini-instruct
**qs/quality:** nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, gemini/gemini-2.5-flash-lite, gemini/gemini-3.5-flash
**qs/speed:** gemini/gemini-2.5-flash-lite, gemini/gemini-3.5-flash, gemini/gemma-4-31b-it, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemma-4-26b-a4b-it
**task/analysis:** gemini/gemma-4-31b-it, gemini/gemma-4-26b-a4b-it, gemini/gemini-2.5-flash-lite, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemini-3-flash-preview
**task/creative:** gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, gemini/gemini-2.5-flash-lite, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemini-3.5-flash
**task/generation:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-31b-it, gemini/gemini-3.5-flash, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemma-4-26b-a4b-it
**task/lookup:** gemini/gemini-2.5-flash-lite, gemini/gemini-3-flash-preview, gemini/gemini-3.5-flash, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it
**task/reasoning:** gemini/gemma-4-26b-a4b-it, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemma-4-31b-it, gemini/gemini-2.5-flash-lite, gemini/gemini-3.5-flash
**task/refactoring:** gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it, gemini/gemini-3.5-flash, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemini-2.5-flash-lite
**task/summarization:** gemini/gemini-2.5-flash-lite, gemini/gemma-4-31b-it, gemini/gemma-4-26b-a4b-it, nvidia_nim/microsoft/phi-4-mini-instruct, gemini/gemini-3.5-flash
**task/translation:** gemini/gemini-2.5-flash-lite, gemini/gemini-3-flash-preview, gemini/gemini-3.5-flash, gemini/gemma-4-26b-a4b-it, gemini/gemma-4-31b-it

## Proficiencies & Deficiencies

**gemini/gemini-2.5-flash-lite**
- Top: task/generation (1.00), task/summarization (1.00), domain/code (1.00)
- Low: qs/quality (0.40), task/refactoring (0.25), domain/business (0.00)

**gemini/gemini-3-flash-preview**
- Top: task/lookup (0.50), task/translation (0.50), domain/general (0.50)
- Low: domain/code (0.00), qs/quality (0.00), qs/speed (0.00)

**gemini/gemini-3.5-flash**
- Top: qs/speed (0.75), task/generation (0.50), task/refactoring (0.50)
- Low: domain/technical (0.00), domain/legal (0.00), qs/balanced (0.00)

**gemini/gemma-4-26b-a4b-it**
- Top: task/refactoring (1.00), task/creative (1.00), task/reasoning (1.00)
- Low: domain/business (0.33), task/generation (0.25), qs/speed (0.25)

**gemini/gemma-4-31b-it**
- Top: task/analysis (1.00), domain/legal (1.00), domain/business (1.00)
- Low: domain/general (0.50), qs/speed (0.50), domain/technical (0.40)

**nvidia_nim/microsoft/phi-4-mini-instruct**
- Top: domain/technical (1.00), qs/quality (1.00), task/reasoning (0.80)
- Low: task/analysis (0.40), domain/code (0.20), qs/balanced (0.20)
