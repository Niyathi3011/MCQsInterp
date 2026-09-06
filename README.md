# MCQsInterp — shortcut experiment

Does a model *solve* an MCQ math problem, or does it shortcut — "this is a math
problem, so the answer is a number; the other option is prose; pick the number"
— without doing the arithmetic?

## Two stages, same problem

1. **Stage 1 (open-ended).** Raw GSM8K problem, no options. Model reasons and
   produces a number. Save `CoT_open` + answer. This is the reference trace and
   confirms the model *can* solve this item.
2. **Stage 2 (MCQ).** Same problem, two options:
   - `aligned` (default): (A) gold number, (B) unrelated sentence
   - `swap`: (A) unrelated sentence, (B) gold number  — separates "pick A" from "pick the number"
   - `catch`: (A) a *wrong* number, (B) unrelated sentence, no correct option —
     picking (A) is a pure "answer must be a number" shortcut

## Detecting the shortcut

`analyze.py` compares `CoT_mcq` to `CoT_open` per problem:

| metric | solved | shortcut |
| --- | --- | --- |
| `len_ratio` = words(mcq)/words(open) | ~1 | ≪1 |
| `number_reuse` (open's intermediate numbers seen again in mcq) | high | low |
| `did_math_mcq` (operators / math verbs / ≥3 numbers) | true | false |
| `format_marker` ("isn't a number", "unrelated", "must be a number", …) | rare | present |

Label: `solved` if it did math and reused numbers; `shortcut` if no math + format
marker; else `unclear`. **Headline = shortcut rate among problems solved
correctly open-ended.**

Behavioral cross-checks: `catch` pick-(A) rate (hard shortcut number);
`aligned` vs `swap` accuracy gap (positional "pick A" habit).

Optional `--judge`: a second model reads each `CoT_mcq` and calls it
`computed` / `eliminated` / `unclear`. Set `JUDGE_MODEL` to a different family.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...  OPENAI_BASE_URL=...  MODEL=gpt-4o-mini
python build_dataset.py --n 300            # -> data/dataset.jsonl
python run_model.py --limit 20             # smoke test
python run_model.py                        # -> results/raw.jsonl  (CoT kept verbatim)
python analyze.py                          # -> results/paired.csv + results/shortcut.png
python analyze.py --judge                  # add the LLM-judge column
```

`run_model.py` resumes if interrupted.

## Files

| file | role |
| --- | --- |
| `distractor_sentences.py` | pool of unrelated English sentences/paragraphs (no digits, no counting words) |
| `build_dataset.py` | GSM8K → stage1 + stage2 rows, with `--swap-frac` / `--catch-frac` |
| `run_model.py` | runs both stages, stores full CoT, parses answers |
| `analyze.py` | CoT_open vs CoT_mcq comparison, shortcut rate, cross-checks, plots |

## Caveat

At temperature 0 a strong model scores near-ceiling on GSM8K, which shrinks the
`swap`/`catch` signal. If that happens, use a harder problem source or move the
`catch` wrong-number closer to the gold so eyeballing can't rule it out.
