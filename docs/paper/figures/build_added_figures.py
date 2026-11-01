"""Redraw report figures from saved results, without rerunning experiments.
Run with PYTHONNOUSERSITE=1 python3 docs/paper/figures/build_added_figures.py.
"""
from pathlib import Path
import ast
import json
import re
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
# F4: preserve the supplied graph exactly.
shutil.copyfile(ROOT/'temp_results/05_single_session_test__finetuned_vs_untuned/scaling_test.png', OUT/'scaling_test.png')
# F5: original numerical x positions, with staggered labels to separate 3B / Qwen3-4B.
model_sizes = next(ast.literal_eval(n.value) for n in ast.parse((ROOT/'extra/_shared.py').read_text()).body
                   if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODEL_SIZES' for t in n.targets))
pvi_source = ROOT/'temp_results/03_information_in_thoughts__PVI/pvi_report_with_qwen3_4B.txt'
pvi = {}
for line in pvi_source.read_text().splitlines():
    match = re.match(r'\| (0\.5B|3B|7B|Qwen3-4B) \| 2608 \| ([\d.]+) \[([\d.]+), ([\d.]+)\]',line)
    if match: pvi[match[1]] = list(map(float,match.groups()[1:]))
assert set(pvi)=={'0.5B','3B','7B','Qwen3-4B'}
fig, ax = plt.subplots(figsize=(7.4,4.2))
for names, color, label in [(['0.5B','3B','7B'],'#2166ac','Qwen2.5'),(['Qwen3-4B'],'#d95f02','Qwen3 control')]:
    x=[model_sizes[n]['params']/1e9 for n in names]; y=[pvi[n][0] for n in names]
    err=[[pvi[n][0]-pvi[n][1] for n in names],[pvi[n][2]-pvi[n][0] for n in names]]
    ax.errorbar(x,y,yerr=err,fmt='o-' if len(names)>1 else 'D',color=color,linewidth=1.7,capsize=4,markersize=6,label=label)
ax.set_xscale('log');ax.set_xlim(.30,8.3);ax.set_ylim(2.7,9.8)
order=['0.5B','3B','Qwen3-4B','7B']
ax.set_xticks([model_sizes[n]['params']/1e9 for n in order])
ax.set_xticklabels(['0.36B\n(Qwen2.5-0.5B)','2.77B\n(Qwen2.5-3B)','\n\n3.60B\n(Qwen3-4B)','6.53B\n(Qwen2.5-7B)'],fontsize=8)
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Non-embedding parameters (log scale)',labelpad=9)
ax.set_ylabel('PVI (bits per turn)');ax.set_title('Information in Analysis and Strategy')
ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,loc='upper right',fontsize=9)
fig.subplots_adjust(left=.10,right=.98,top=.90,bottom=.29)
for ext in ['pdf','png']:fig.savefig(OUT/f'pvi_by_size_with_qwen3_4B.{ext}',dpi=240)
plt.close(fig)
# F7: extract only the requested rows from the saved session-level summary.
session_source=ROOT/'temp_results/11_multi_session__all_7B_systems/all_7B_systems.txt'
metrics={'Success (0-1':'Success','IP (0-1':'IP','PRI observed (0-1':'PRI'}
series={m:{} for m in metrics.values()}; current=None
for line in session_source.read_text().splitlines():
    for prefix,metric in metrics.items():
        if line.startswith(prefix):current=metric
    if current:
        pairs=re.findall(r'(\d+\.\d+) \(n=(\d+)\)',line)
        if pairs:
            name=re.split(r'\s{2,}',line)[0]
            series[current][name]={'values':[float(v) for v,n in pairs], 'n':[int(n) for v,n in pairs]}
curves=[('fine-tuned 7B, with thoughts, no memory','FT: no memory','#2676c4','-','o'),
        ('fine-tuned 7B, with thoughts + memory','FT: need-state memory','#e56b2f','-','s'),
]
diamonds=[('fine-tuned 7B, with thoughts, single session','FT: with thoughts (S1 only)','#123f82'),
          ('fine-tuned 7B, without thoughts, single session','FT: without thoughts (S1 only)','#70a9df'),
          ('untuned 7B, one call (base_instruct)','Untuned: one call (S1 only)','#555555')]
fig,axes=plt.subplots(1,3,figsize=(11.0,4.4))
for ax,metric in zip(axes,['Success','IP','PRI']):
    for key,label,color,style,marker in curves:
        values=series[metric][key]['values'];assert len(values)==4
        ax.plot([1,2,3,4],values,label=label,color=color,linestyle=style,marker=marker,linewidth=1.8,markersize=5,zorder=3)
    for key,label,color in diamonds:
        values=series[metric][key]['values'];assert len(values)==1
        ax.plot([1],values,marker='D',color=color,linestyle='none',markersize=7,markeredgecolor='white',markeredgewidth=.5,zorder=4)
    ax.set_title(metric if metric!='PRI' else 'PRI (observed)');ax.set_xticks([1,2,3,4],['S1','S2','S3','S4'])
    ax.set_xlim(.83,4.15);ax.set_xlabel('Session');ax.grid(axis='y',alpha=.20);ax.tick_params(labelsize=9)
axes[0].set_ylabel('Score (0-1)')
handles=[Line2D([0],[0],color=c,linestyle=ls,marker=m,linewidth=1.8,label=l) for k,l,c,ls,m in curves]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.115),ncol=2,frameon=False,fontsize=9)
handles=[Line2D([0],[0],color=c,linestyle='none',marker='D',label=l) for k,l,c in diamonds]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=8.5)
fig.subplots_adjust(left=.06,right=.99,top=.91,bottom=.35,wspace=.27)
for ext in ['pdf','png']:fig.savefig(OUT/f'all_7B_systems_by_session.{ext}',dpi=240)
plt.close(fig)
# Sidecar records the plotted values for verification.
selected={m:{k:series[m][k] for k,*_ in curves+diamonds} for m in series}
(OUT/'added_figures_data.json').write_text(json.dumps({'pvi':pvi,'by_session':selected},indent=2)+'\n')
print('Generated three requested figures from saved sources.')
