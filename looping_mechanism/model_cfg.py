"""
Model choice for the analysis scripts written for R1 (model_stop_points, answer_probe,
live_controller): set LOOP_MODEL=Qwen/Qwen3-4B-Thinking-2507 to run them on Qwen3.

  stop / control   the model's stop MLPs and control MLP, found from its own MATH
                   catch-vs-aligned pairs (stop_signal.py --components auto)
  ref_pairs        the pairs file its catch donor is computed from
  probe_layers     residual layers the answer probe reads; probe_layer = the one the
                   stopping rule uses (Qwen3: same relative depth as R1's 14/20/24/27 of 28)
"""
import os

R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
Q3 = "Qwen/Qwen3-4B-Thinking-2507"
MODELS = {
    R1: dict(tag="r1-distill-qwen-7b", stop=[27, 26, 25, 24, 22, 20], control=[15],
             ref_pairs="pairs.jsonl", probe_layers="14,20,24,27", probe_layer=24),
    Q3: dict(tag="qwen3-4b-thinking-2507", stop=[28, 31, 32, 33, 34, 35], control=[18],
             ref_pairs="pairs_qwen3-4b-thinking-2507.jsonl", probe_layers="18,26,31,35", probe_layer=31),
}
MODEL = os.environ.get("LOOP_MODEL", R1)
CFG = MODELS[MODEL]
TAG = CFG["tag"]
