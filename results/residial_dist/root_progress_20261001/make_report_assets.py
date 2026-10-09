"""CPU-only reaggregation for the root-candidate progress report.

No inference, refitting, or new policy selection. Preserve all September files.
Fixed allocation means exactly three candidates at each of the five positions,
including bonus; it is not a uniform weight followed by global top-15.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'training_free'
TEMPS = [1., .7, .5]
REPS = 4000


def interval(point, samples):
    return dict(mean=float(point), ci95=np.quantile(samples, [.025, .975]).tolist())


def plot_original_position(result):
    """All three token scores, with the original hazard and top-M norm."""
    names = ['residual__topm__original', 'proxy__topm__original',
             'complement_power1__topm__original']
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axs = plt.subplots(1, 3, figsize=(11.4, 4.2),
                                layout='constrained', sharey=True)
        for ax, t in zip(axs, TEMPS):
            cells = result[str(t)]['cells']
            values = np.array([100*cells[k]['mean'] for k in names])
            ci = np.array([cells[k]['ci95'] for k in names])*100
            ax.bar(np.arange(3), values, .65,
                   color=['#be5046', '#3479ae', '#338266'])
            ax.errorbar(np.arange(3), values,
                        yerr=[values-ci[:, 0], ci[:, 1]-values],
                        fmt='none', ecolor='#333333', capsize=3, lw=.8)
            for x, yy, hi in zip(np.arange(3), values, ci[:, 1]):
                ax.text(x, hi+1.5, f'{yy:.3f}', ha='center', fontsize=9)
            ax.set_xticks(np.arange(3), ['Residual', 'Proxy', 'e(1-q)'])
            ax.set_title(f'T = {t:g}')
            ax.set_ylim(0, 102)
            ax.grid(axis='y', alpha=.17)
            ax.set_axisbelow(True)
        axs[0].set_ylabel('Expected root coverage (%)')
        fig.supxlabel('Original position formula + top-M normalization for all scores | P1 off | 96 prompts, 15 roots\n'
                      'Saved top-15 replay, not live cache hit | pointwise 95% prompt CI', fontsize=9)
        for ext in ['png', 'pdf']:
            fig.savefig(HERE/f'06_token_original_position.{ext}', dpi=180)
        plt.close(fig)


def main():
    HERE.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'figure.dpi': 180})
    prior = {(float(r['T']), r['policy']): float(r['coverage'])
             for r in csv.DictReader((SOURCE/'confirmation_policies.csv').open())}
    result = {}; hashes = {}; csv_rows = []
    for t in TEMPS:
        path = SOURCE / f'confirmation/mixed_t{t:g}_seed271.npz'
        hashes[str(path.relative_to(HERE.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            h = z['row::h'].astype(np.float64)
            if h.shape[1] != 5 or not np.allclose(h.sum(-1), 1., atol=2e-6):
                raise ValueError('Requires K=4 plus bonus and normalized true hazard')
            values = {}
            for source in ['residual', 'proxy', 'complement_power1']:
                local = z[f'local::{source}'].astype(np.float64)
                values[f'{source}__fixed3'] = (h*local).sum(-1)
                for norm in ['topm', 'full']:
                    for weight in ['original', 'expected_rejection', 'expected_mix0.25']:
                        name = f'{source}__{norm}__{weight}'
                        values[name] = z[f'policy::{name}'].astype(np.float64)
            dataset, prompt = z['dataset'], z['prompt_id']
            names = list(values)
            matrix = np.column_stack([values[name] for name in names])
            if not np.isfinite(matrix).all() or matrix.min() < -1e-6 or matrix.max() > 1+1e-6:
                raise ValueError('Invalid coverage')
            groups = {}
            for ds in np.unique(dataset):
                groups[str(ds)] = np.stack([
                    matrix[(dataset == ds) & (prompt == pid)].mean(0)
                    for pid in np.unique(prompt[dataset == ds])])
            if len(groups) != 4 or any(len(a) != 24 for a in groups.values()):
                raise ValueError('Expected 96 prompts, four equal task groups')
        point = np.mean([a.mean(0) for a in groups.values()], axis=0)
        rng = np.random.default_rng(20261001)
        bootstrap = np.mean([a[rng.integers(len(a), size=(REPS, len(a)))].mean(1)
                             for a in groups.values()], axis=0)
        cells = {}; csv_rounding_error = 0.
        for i, name in enumerate(names):
            cells[name] = interval(point[i], bootstrap[:, i])
            cells[name]['dataset_means'] = {ds: float(a[:, i].mean()) for ds, a in groups.items()}
            if not name.endswith('__fixed3'):
                csv_rounding_error = max(csv_rounding_error, abs(point[i]-prior[t, 'policy::'+name]))
            csv_rows.append(dict(T=t, policy=name, coverage=point[i],
                                 low=cells[name]['ci95'][0], high=cells[name]['ci95'][1]))
        if csv_rounding_error > 1e-12:
            raise ValueError(('Prior CSV changed', t, csv_rounding_error))
        pairs = {
            'residual_minus_proxy_fixed': ('residual__fixed3', 'proxy__fixed3'),
            'residual_minus_proxy_dynamic': ('residual__topm__original', 'proxy__topm__original'),
            'allocation_gain_residual': ('residual__topm__original', 'residual__fixed3'),
            'allocation_gain_proxy': ('proxy__topm__original', 'proxy__fixed3'),
            'duet_residual_dynamic_minus_proxy_fixed': ('residual__topm__original', 'proxy__fixed3'),
            'hbar_minus_original': ('proxy__topm__expected_rejection', 'proxy__topm__original'),
            'mix_minus_original': ('proxy__topm__expected_mix0.25', 'proxy__topm__original'),
            'mix_minus_hbar': ('proxy__topm__expected_mix0.25', 'proxy__topm__expected_rejection'),
            'eq_source_only_minus_proxy': ('complement_power1__topm__original', 'proxy__topm__original'),
            'eq_same_improved_allocation_minus_proxy': ('complement_power1__full__expected_mix0.25', 'proxy__full__expected_mix0.25'),
        }
        contrasts = {}
        for label, (first, second) in pairs.items():
            a, b = names.index(first), names.index(second)
            contrasts[label] = dict(first=first, second=second,
                                   **interval(point[a]-point[b], bootstrap[:, a]-bootstrap[:, b]))
        result[str(t)] = dict(prompts=96, sampled_steps=len(matrix), cells=cells,
                              contrasts=contrasts, prior_csv_max_error=csv_rounding_error)
        print('T', t, '2x2', {k: round(100*cells[k]['mean'], 4) for k in
              ['residual__fixed3', 'residual__topm__original', 'proxy__fixed3', 'proxy__topm__original']}, flush=True)

    (HERE/'summary.json').write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(), temperatures=result,
        source_sha256=hashes, bootstrap_replicates=REPS, bootstrap_seed=20261001,
        metric='Expected correction/bonus coverage on saved P1-off trajectories; not observed cache hit',
        fixed_allocation='Exactly 3 candidates per each of K+1=5 positions, total 15',
        dynamic_allocation='Stored frozen float32 replay: top-17 local pools, top-M normalization, global top-15',
        limitations=['New aggregation of existing data, no new model inference.',
                    'Pointwise paired task-stratified prompt bootstrap; exploratory contrasts, no multiple-testing correction.',
                    'Ties use saved replay outcomes, not original GPU 17-slot wire prefix.',
                    'All sampling temperatures share 96 prompt texts; not 288 independent prompts.',
                    'P1 scheduling, root readiness and actual cache hits not inferred from coverage.']), indent=2)+'\n')
    with (HERE/'tables.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)

    # Factorial figure: all four cells have the same 15-root budget.
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 4.2), layout='constrained', sharey=True)
    for ax, t in zip(axs, TEMPS):
        cells=result[str(t)]['cells']; x=np.arange(2)
        for offset, source, color, label in [(-.18,'residual','#be5046','Residual'),(.18,'proxy','#3479ae','Proxy')]:
            names=[source+'__fixed3',source+'__topm__original']
            y=np.array([100*cells[k]['mean'] for k in names])
            ci=np.array([cells[k]['ci95'] for k in names])*100
            ax.bar(x+offset,y,.33,color=color,label=label)
            ax.errorbar(x+offset,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='none',ecolor='#333333',capsize=3,lw=.8)
            for xx,yy in zip(x+offset,y):ax.text(xx,yy+3.0,f'{yy:.2f}',ha='center',fontsize=9)
        ax.set_xticks(x,['Fixed 3 / position','Original allocation']);ax.set_title(f'T = {t:g}')
        ax.set_ylim(0,104);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    axs[0].set_ylabel('Expected root coverage (%)');axs[0].legend(frameon=False,loc='upper left',fontsize=9)
    fig.supxlabel('P1 off | 96 prompts per T | 15 roots | saved top-15 replay | pointwise 95% prompt CI',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(HERE/f'01_factorial.{ext}')
    plt.close(fig)

    fig, axs=plt.subplots(1,3,figsize=(11.4,4.2),layout='constrained',sharey=True)
    names=['proxy__topm__original','proxy__topm__expected_rejection','proxy__topm__expected_mix0.25']
    for ax,t in zip(axs,TEMPS):
        cells=result[str(t)]['cells'];y=np.array([100*cells[k]['mean'] for k in names])
        ci=np.array([cells[k]['ci95'] for k in names])*100
        ax.bar(np.arange(3),y,.65,color=['#777777','#d29435','#338266'])
        ax.errorbar(np.arange(3),y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='none',ecolor='#333333',capsize=3,lw=.8)
        for x,yy in enumerate(y):ax.text(x,yy+3.,f'{yy:.2f}',ha='center',fontsize=9)
        ax.set_xticks(np.arange(3),['Original','Overlap only','25% mix'],rotation=12)
        ax.set_ylim(0,104);ax.set_title(f'T = {t:g}');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    axs[0].set_ylabel('Expected root coverage (%)')
    fig.supxlabel('Only position weights change | source = e | top-M normalization fixed | saved replay, not live cache hit',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(HERE/f'02_position.{ext}')
    plt.close(fig)

    # Existing actual-wire check: T=.7 is the arm whose frozen score is e(1-q).
    wire=json.loads((SOURCE/'runtime_snapshot_checks.json').read_text())['summaries']['0.7']
    labels=['Proxy\noriginal allocation','Proxy\nimproved allocation','e(1-q)\nimproved allocation']
    keys=['original_proxy','identical_allocation_proxy','overall']
    fig,ax=plt.subplots(figsize=(6.4,4),layout='constrained')
    y=np.array([100*wire[k]['coverage']['mean'] for k in keys])
    ci=np.array([wire[k]['coverage']['ci95'] for k in keys])*100
    ax.bar(np.arange(3),y,.65,color=['#777777','#3479ae','#338266'])
    ax.errorbar(np.arange(3),y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='none',ecolor='#333333',capsize=4)
    for x,yy in enumerate(y):ax.text(x,yy+2.5,f'{yy:.3f}',ha='center')
    ax.set_xticks(np.arange(3),labels);ax.set_ylim(0,101);ax.set_ylabel('Expected root coverage (%)')
    ax.set_title('T = 0.7 | 17-slot wire, first 15 roots')
    delta=wire['overall']['vs_identical_allocation_proxy']
    fig.supxlabel('Score-only gain: '+f'{100*delta["mean"]:+.3f} pp; 95% CI [{100*delta["ci95"][0]:+.3f}, {100*delta["ci95"][1]:+.3f}]'
                  +'\nReconstructed float32 logits; P1 off; not live cache hit',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(HERE/f'03_token_wire.{ext}')
    plt.close(fig)
    plot_original_position(result)


if __name__ == '__main__':
    main()
