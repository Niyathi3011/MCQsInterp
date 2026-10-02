# NAACL short paper draft

`main.tex` (ACL template, anonymous review mode), `references.bib`, `figures/`.

Build (no TeX install needed; Tectonic fetches packages on demand):

```bash
python paper/naacl_short/make_figures.py          # figures/stop_gate.pdf from the stored results
cd paper/naacl_short && /workspace/tools/tectonic -X compile main.tex
```

Red `[TODO: ...]` marks results that are still running (AQuA without the correct
option, intervention seeds 2-3, second model). Every number in the text comes from
`looping_mechanism/RESULTS.md`.
