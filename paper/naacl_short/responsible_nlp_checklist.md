# ARR Responsible NLP Checklist: draft answers

For: *Stopping Is Not Resolving: Why Reasoning Models Loop When No Answer Fits* (ARR October 2026, short paper).
Copy into the OpenReview checklist form. Items marked **VERIFY** need the authors' confirmation.

## A. For every submission

**A1. Did you describe the limitations of your work?**
Yes. Section "Limitations", after the conclusion.

**A2. Did you discuss any potential risks of your work?**
Yes, briefly. The work is an analysis of model behaviour on public benchmarks, with no new model, data
about people, or deployment. The main risk is over-reading the interventions: we show that forcing a
stop does not resolve loops, and that allowing "None" works only for some models. Limitations states
that results come from 4–8B models with a 12k-token budget and may not transfer.
(Optional: add one sentence to Limitations saying that loop-breaking methods evaluated only on whether
the loop ends can overstate their benefit.)

**A3. Do the abstract and introduction summarize the paper's main claims?**
Yes. Abstract and the contribution list in Section 1.

## B. Did you use or create scientific artifacts?

Yes: public benchmarks, public models, and open-source software. We create derived evaluation items
(two-option and correct-option-removed versions of existing questions).

**B1. Did you cite the creators of artifacts you used?**
Yes: GSM8K (Cobbe et al.), MATH / MATH-500 (Hendrycks et al.), AQuA (Ling et al.), LogiQA (Liu et al.),
DeepSeek-R1 distills (DeepSeek-AI), Qwen3 (Yang et al.), vLLM (Kwon et al.), TransformerLens (Nanda and
Bloom), logit lens (nostalgebraist). Section 2 and References.

**B2. Did you discuss the license or terms for use and/or distribution of any artifacts?**
Answer: No in the paper (space). Licenses, **VERIFY each before submitting**:
- GSM8K: MIT
- MATH / MATH-500: MIT
- AQuA-RAT: Apache 2.0
- LogiQA: **VERIFY** (released for research use; check the dataset repository)
- DeepSeek-R1-Distill-Qwen-7B: MIT (base Qwen2.5: Apache 2.0)
- DeepSeek-R1-Distill-Llama-8B: MIT, plus the Llama 3.1 Community License for the base
- Qwen3-4B-Thinking-2507: Apache 2.0
- vLLM: Apache 2.0. TransformerLens: MIT. math-verify: Apache 2.0.

**B3. Did you discuss if your use of existing artifacts was consistent with their intended use?**
Yes, implicitly: all artifacts are research benchmarks and research models, used for evaluation and
analysis only. The derived items (correct option removed, two-option versions) are for analysis and
will be released under the source datasets' terms.

**B4. Did you discuss the steps taken to check whether the data contains personally identifying info or
offensive content?**
N/A. Math word problems and logical-reasoning questions from established public benchmarks; no
personal data was collected.

**B5. Did you provide documentation of the artifacts?**
Yes. Section 2 describes the datasets, the derived conditions and their sizes; the code release
documents the construction (build scripts).

**B6. Did you report relevant statistics like the number of examples, details of train/test splits?**
Yes. Section 2 and Table 1: 150 MATH, 200 GSM8K, 227 AQuA, 200 LogiQA controlled items; natural sets of
500 / 500 / 254 / 500. Held-out design (questions that loop naturally are test; learning pairs come from
the other questions) in Section 2; sample sizes in Section 4 and the table captions.

## C. Did you run computational experiments?

Yes.

**C1. Did you report the number of parameters in the models used, the total computational budget, and the
computing infrastructure used?**
Parameters: yes (4B, 7B, 8B; model names in Section 2).
Compute: add to the appendix if space allows: *one NVIDIA RTX 3090 (24 GB); roughly 350 GPU-hours in
total for generation (vLLM) and analysis (TransformerLens)*. **VERIFY the estimate.**

**C2. Did you discuss the experimental setup, including hyperparameter search and best-found values?**
Yes. Section 2: T=0.6, top-p 0.95, 12k-token budget, 4 seeds (Llama: 2), greedy decoding for R1; answer
budget 1.2k tokens; live-test trigger at 6k thinking tokens, fixed in advance (no hyperparameter search).

**C3. Did you report descriptive statistics about your results?**
Yes. Means over seeds; 95% bootstrap confidence intervals over questions for the live test (Table 2
caption) and the healthy-vs-loop comparison (Section 4); counts and sample sizes throughout.

**C4. If you used existing packages, did you report the implementation, model, and parameter settings used?**
Yes: vLLM, TransformerLens, math-verify are cited and named. Versions (add to the appendix if space
allows): vLLM 0.28.0, TransformerLens 3.8.1, transformers 5.16.1, PyTorch 2.13.0, math-verify 0.9.0,
scikit-learn 1.9.0.

## D. Did you use human annotators (e.g., crowdworkers) or research with human participants?

No. No crowdworkers or study participants.
**VERIFY:** the paper says commit points were "hand-checked". That check was done on random samples during
the analysis with the AI assistant (see E1). If you want the paper's wording to hold, spot-check about 20
commit points yourself (they are in `looping_mechanism/data/commit_point_handcheck.md` and the
`test_loops.jsonl` files); otherwise reword to "checked on random samples".

## E. Did you use AI assistants (e.g., ChatGPT, Copilot) in your research, coding, or writing?

**E1. Yes. VERIFY and adjust to what you did.** An AI coding assistant (Claude, Anthropic) was used to write
and run experiment code, compute and summarise results, and draft and edit the paper text. The authors
designed the study, reviewed all code, results and text, and take responsibility for the content.
