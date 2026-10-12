"""Export full-input comparisons and separate target/draft diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

R = Path(__file__).resolve().parent
OUT = R / 'figs'
OUT.mkdir(exist_ok=True)
data = json.loads((R / 'ANALYSIS.json').read_text())
components = json.loads((R / 'COMPONENTS.json').read_text())['arms']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
colors = ['#778899', '#147d92']

with PdfPages(OUT / 'REVIEW.pdf') as pdf:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    rows = data['comparisons']
    x = np.arange(len(rows))
    for ax, key, title in zip(axes, ['tps', 'verify_ms', 'al'],
                              ['Decode throughput (token/s)', 'Verification (ms)', 'AL (including recovery)']):
        for arm, name in enumerate(['Paper', 'MLSys ver0.1']):
            values = [r['baseline_summary' if arm == 0 else 'candidate_summary'][key] for r in rows]
            bars = ax.bar(x + (arm - .5) * .36, values, .36, label=name, color=colors[arm])
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
        ax.set_xticks(x, ['Tree repeat 0', 'Tree repeat 1', 'Chain'])
        ax.set_title(title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.13)
        ax.grid(axis='y', alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle('Actual paper checkout versus MLSys ver0.1 | B=1, T=1, 480 inputs per run')
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig.savefig(OUT / '01_full_comparison.png', dpi=180)
    pdf.savefig(fig)
    plt.close(fig)

    fields = [('target_pre_ms', 'Target before exit'), ('target_post_ms', 'Target after exit'),
              ('target_tree_walk_ms', 'Tree acceptance'), ('draft_glue_ms', 'Draft glue'),
              ('draft_p1_replay_ms', 'Draft P1 graph'), ('draft_p2_replay_ms', 'Draft P2 graph'),
              ('draft_p1_total_ms', 'P1 phase envelope'), ('draft_p2_total_ms', 'P2 phase envelope')]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (key, title) in zip(axes.flat, fields):
        for arm, name in enumerate(['paper', 'ver01']):
            vals = []
            for repeat in ('tree_r0', 'tree_r1'):
                s = components[repeat + '_' + name + '_profile']['summary'].get(key)
                vals.append(s['median'] if s else np.nan)
            bars = ax.bar(np.arange(2) + (arm - .5) * .35, vals, .35, color=colors[arm], label=name)
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
        ax.set_xticks([0, 1], ['Repeat 0', 'Repeat 1'])
        ax.set_title(title)
        ax.set_ylabel('Median span (ms)')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
        ax.grid(axis='y', alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle('Separate 24-input diagnostics | overlapping and nested spans are not additive')
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / '02_components.png', dpi=180)
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    fields_chain = [('target_pre_ms', 'Target before exit'), ('target_post_ms', 'Target after exit'),
                    ('target_chain_accept_ms', 'Chain acceptance'), ('draft_glue_ms', 'Draft glue'),
                    ('draft_p1_total_ms', 'P1 phase envelope'), ('draft_p2_total_ms', 'P2 phase envelope')]
    for ax, (key, title) in zip(axes.flat, fields_chain):
        values = [components['chain_'+arm+'_profile']['summary'][key]['median'] for arm in ('paper', 'ver01')]
        bars = ax.bar([0, 1], values, color=colors)
        ax.bar_label(bars, fmt='%.3f', padding=2)
        ax.set_xticks([0, 1], ['Paper', 'MLSys ver0.1'])
        ax.set_title(title)
        ax.set_ylabel('Median span (ms)')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    fig.suptitle('Chain component diagnostics | separate 24-input runs, B=1, T=1')
    fig.tight_layout()
    fig.savefig(OUT / '04_chain_components.png', dpi=180)
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8))
    for ax, name in zip(axes, ['tree_r0_paper_profile', 'tree_r0_ver01_profile']):
        samples = [v for v in components[name]['timeline'] if v['source'] == 2 and
                   v['target_pre_start'] is not None and v['target_post_start'] is not None]
        samples.sort(key=lambda v: v['target_final'] - v['target_pre_start'])
        row = samples[len(samples)//2]
        origin = min(row['target_pre_start'], row['p1_start'])
        for label, start, end, y, color in [
                ('Target before exit', 'target_pre_start', 'target_pre_end', 1, '#6a8caf'),
                ('Target after exit', 'target_post_start', 'target_post_end', 1, '#3b5976'),
                ('P1 phase envelope', 'p1_start', 'p1_end', 0, '#3c9d9b'),
                ('P2 phase envelope', 'p2_start', 'p2_end', 0, '#e4b04a')]:
            ax.broken_barh([((row[start]-origin)/1e6, (row[end]-row[start])/1e6)],
                           (y-.3, .6), facecolors=color, label=label)
        for key, label, color in [('proxy_ready_estimate', 'Proxy enqueue', '#b94362'),
                                  ('target_final', 'Final logits', '#444444')]:
            ax.axvline((row[key]-origin)/1e6, color=color, linestyle='--', linewidth=1, label=label)
        ax.set_yticks([0, 1], ['Draft GPU', 'Target GPU'])
        ax.set_xlabel('Approximate CPU-anchored time (ms)')
        label = 'Paper' if name == 'tree_r0_paper_profile' else 'MLSys ver0.1'
        ax.set_title(f'{label} | measured prompt {row["group"]}, step {row["step"]} | P2 cache hit')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8, loc='lower center')
    fig.suptitle('Observed timelines | separate process runs, approximate cross-device alignment', y=.99)
    fig.tight_layout(rect=(0, .12, 1, .96))
    fig.savefig(OUT / '03_timelines.png', dpi=180, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(OUT / 'REVIEW.pdf')
