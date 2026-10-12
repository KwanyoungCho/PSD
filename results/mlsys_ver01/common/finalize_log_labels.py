"""Apply only after timing campaigns: make legacy primary log labels use the population."""
from pathlib import Path
R=Path(__file__).resolve().parents[3]
p=R/'ssd/ssd/engine/llm_engine.py';s=p.read_text()
s=s.replace('''            print(
                f"[raw metrics] Avg Tokens per step (incl recovery): {avg_tokens_per_step:.2f}", flush=True)''','''            selected_al = f"{selected['al']:.2f}" if selected['al'] is not None else "N/A"
            print(f"[metrics] Avg Tokens per step (incl recovery): {selected_al}", flush=True)
            print(f"[raw metrics] All-event accepted length: {avg_tokens_per_step:.2f}", flush=True)''')
s=s.replace('''                print(
                    f"[metrics] Avg Cache Hits: {sum(METRICS['cache_hits']) / len(METRICS['cache_hits']):.2f}", flush=True)''','''                selected_hit = f"{selected['cache_hit']:.2f}" if selected['cache_hit'] is not None else "N/A"
                print(f"[metrics] Avg Cache Hits: {selected_hit}", flush=True)
                print(f"[raw metrics] All-event cache hit: {sum(METRICS['cache_hits']) / len(METRICS['cache_hits']):.2f}", flush=True)''')
p.write_text(s)
