"""Export confirmation figures and readable tables from measured artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def main():
    result=json.loads((HERE/'confirmation_summary.json').read_text())['temperatures']
    diag=json.loads((HERE/'diagnostics.json').read_text())
    temps=[1.,.7,.5];colors=['#C44E52','#777777','#4C72B0','#228833']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'figure.dpi':160})
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    names=['original_residual','original_proxy','proxy_allocation','overall']
    labels=['Original residual','Original proxy score','Proxy score, same allocation','Frozen draft-aware rule']
    x=np.arange(3);width=.19
    for j,(name,label,color) in enumerate(zip(names,labels,colors)):
        y=[100*(diag[str(t)]['ablation']['allocation_only']['coverage']['mean'] if name=='proxy_allocation'
                else result[str(t)]['selected'][name]['coverage']['mean']) for t in temps]
        axs[0].bar(x+(j-1.5)*width,y,width,label=label,color=color)
    axs[0].set_xticks(x,[str(t) for t in temps]);axs[0].set_xlabel('Actual generation temperature')
    axs[0].set_ylabel('Expected root coverage (%)');axs[0].set_ylim(0,115)
    axs[0].set_title('96 unseen prompts at each temperature')
    axs[0].legend(frameon=False,fontsize=8,loc='upper left')
    for j,(field,label,color) in enumerate([
        ('vs_original_proxy','vs original proxy score','#228833'),
        ('vs_identical_proxy_allocation','vs proxy, identical allocation','#4C72B0')]):
        data=[(diag[str(t)]['ablation']['selected'][field] if field=='vs_identical_proxy_allocation'
               else result[str(t)]['selected']['overall'][field]) for t in temps]
        y=np.array([v['mean']*100 for v in data]);ci=np.array([v['ci95'] for v in data])*100
        axs[1].errorbar(x+(j-.5)*.13,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),
                        marker='o',linestyle='none',capsize=4,color=color,label=label)
    axs[1].axhline(0,color='#777777',lw=.8);axs[1].set_xticks(x,[str(t) for t in temps])
    axs[1].set_xlabel('Actual generation temperature');axs[1].set_ylabel('Coverage difference (percentage points)')
    axs[1].set_title('Paired prompt bootstrap, pointwise 95% CI');axs[1].legend(frameon=False,fontsize=8)
    fig.supxlabel('Frozen replay: small source-score gains do not survive the wire/tie sensitivity check.',fontsize=8)
    fig.savefig(HERE/'confirmation.png');fig.savefig(HERE/'confirmation.pdf');plt.close(fig)
    lines=['# Independent confirmation tables','',
           'Prompt-balanced coverage; 96 unseen prompts per T. CI: 4,000 paired, dataset-stratified prompt resamples.',
           'The original proxy baseline already uses q in its position weights; “proxy” describes the candidate source score.',
           '', '| T | Policy | Coverage (%) | vs original proxy (pp, 95% CI) | vs optimized proxy (pp, 95% CI) |',
           '|---|---|---:|---:|---:|']
    fmt=lambda d:f'{d["mean"]*100:+.4f} [{d["ci95"][0]*100:+.4f}, {d["ci95"][1]*100:+.4f}]'
    for t in temps:
        for name in names:
            s=result[str(t)]['selected'][name]
            lines.append(f'| {t} | {name} | {100*s["coverage"]["mean"]:.4f} | {fmt(s["vs_original_proxy"])} | {fmt(s["vs_optimized_proxy"])} |')
    lines+=['','## Factorized ablations','', '| T | Change | Coverage (%) | vs original proxy (pp, 95% CI) |',
            '|---|---|---:|---:|']
    for t in temps:
        for name,r in diag[str(t)]['ablation'].items():
            lines.append(f'| {t} | {name} | {100*r["coverage"]["mean"]:.4f} | {fmt(r["vs_base"])} |')
    lines+=['','## Frozen family winners (not reselected on confirmation)','',
            '| T | Family | Frozen key | Coverage (%) | vs original proxy (pp) |','|---|---|---|---:|---:|']
    for t in temps:
        for name,s in result[str(t)]['selected'].items():
            lines.append(f'| {t} | {name} | `{s["policy"]}` | {100*s["coverage"]["mean"]:.4f} | {100*s["vs_original_proxy"]["mean"]:+.4f} |')
    lines+=['','## Dataset means for primary policies','', '| T | Policy | Alpaca | C4 | GSM | HumanEval |',
            '|---|---|---:|---:|---:|---:|']
    for t in temps:
        for name in names:
            d=result[str(t)]['selected'][name]['dataset_means']
            lines.append(f'| {t} | {name} | '+' | '.join(f'{100*d[k]:.3f}' for k in ['alpaca','c4','gsm','humaneval'])+' |')
    (HERE/'TABLES.md').write_text('\n'.join(lines)+'\n')
    # Main robustness figure: same frozen rules, actual 17-slot wire prefix.
    path=HERE/'runtime_snapshot_checks.json'
    if path.exists():
        wire=json.loads(path.read_text())['summaries']
        fig,axs=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
        for j,(name,label,color) in enumerate([
            ('original_proxy','Original proxy score','#777777'),
            ('identical_allocation_proxy','Proxy score, same allocation','#4C72B0'),
            ('overall','Frozen draft-aware rule','#228833')]):
            vals=[100*wire[str(t)][name]['coverage']['mean'] for t in temps]
            axs[0].bar(x+(j-1)*.24,vals,.24,color=color,label=label)
        axs[0].set_xticks(x,[str(t) for t in temps]);axs[0].set_ylim(0,113)
        axs[0].set_xlabel('Actual generation temperature');axs[0].set_ylabel('Expected root coverage (%)')
        axs[0].set_title('17-slot wire, first 15 roots');axs[0].legend(frameon=False,fontsize=8,loc='upper left')
        for j,(field,label,color) in enumerate([
            ('vs_original_proxy','vs original proxy score','#228833'),
            ('vs_identical_allocation_proxy','vs proxy, identical allocation','#4C72B0')]):
            d=[wire[str(t)]['overall'][field] for t in temps]
            y=np.array([v['mean']*100 for v in d]);ci=np.array([v['ci95'] for v in d])*100
            axs[1].errorbar(x+(j-.5)*.13,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),
                marker='o',linestyle='none',capsize=4,color=color,label=label)
        axs[1].axhline(0,color='#777777',lw=.8);axs[1].set_xticks(x,[str(t) for t in temps])
        axs[1].set_xlabel('Actual generation temperature');axs[1].set_ylabel('Coverage difference (percentage points)')
        axs[1].set_title('Paired prompt bootstrap, pointwise 95% CI');axs[1].legend(frameon=False,fontsize=8)
        fig.savefig(HERE/'runtime_confirmation.png');fig.savefig(HERE/'runtime_confirmation.pdf');plt.close(fig)

if __name__=='__main__':main()
