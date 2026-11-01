"""Make the corpus visual from saved counts; no inferred category grouping."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
stats=json.loads((ROOT/'temp_results/01_dataset_quality__ours_vs_ExTES_vs_ESConv/dataset_stats.txt').read_text())
assert sum(stats['problem_types'].values())==stats['profiles']==sum(stats['resistance'].values())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig=plt.figure(figsize=(10,5.8),facecolor='white')
for x,label,value in zip([.14,.38,.62,.86],['Profiles','Kept sessions','Dialogue turns','Annotated turns'],[stats['profiles'],stats['sessions_kept'],stats['turns_per_session']['total'],stats['annotations']]):
    fig.text(x,.94,f'{value:,}',ha='center',fontsize=19,fontweight='bold',color='#184e70')
    fig.text(x,.895,label,ha='center',fontsize=10,color='#505b65')
left=fig.add_axes([.04,.35,.54,.48]);right=fig.add_axes([.64,.35,.31,.48])
items=list(stats['problem_types'].items());palette=list(plt.get_cmap('tab20').colors)
colors=[palette[i] for i in [0,2,4,6,8,10,12,14,16,18,1,5,9]]
left.pie([n for k,n in items],startangle=90,counterclock=False,colors=colors,wedgeprops={'width':.38,'edgecolor':'white','linewidth':1})
left.text(0,.06,'13',ha='center',fontsize=24,fontweight='bold',color='#184e70');left.text(0,-.20,'problem types',ha='center',fontsize=10)
left.set_title('(a) Emotional-support scenarios',fontsize=12,pad=8)
handles=[Patch(facecolor=c,label=f'{k.capitalize()}  {n/10:.1f}%') for (k,n),c in zip(items,colors)]
left.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.01),ncol=2,frameon=False,fontsize=8.2,columnspacing=1.1,handlelength=.9,labelspacing=.55)
rcolors=['#79b8d4','#287b94','#dca05b'];rs=['low','medium','high']
right.pie([stats['resistance'][r] for r in rs],startangle=90,counterclock=False,colors=rcolors,autopct='%1.0f%%',pctdistance=.80,wedgeprops={'width':.38,'edgecolor':'white','linewidth':1},textprops={'fontsize':11})
right.text(0,.05,'1,000',ha='center',fontsize=21,fontweight='bold',color='#184e70');right.text(0,-.20,'profiles',ha='center',fontsize=10)
right.set_title('(b) Resistance levels',fontsize=12,pad=8)
right.legend(handles=[Patch(facecolor=c,label=f'{r.capitalize()}  {stats["resistance"][r]}') for r,c in zip(rs,rcolors)],loc='upper center',bbox_to_anchor=(.5,-.02),frameon=False,ncol=1,fontsize=10,labelspacing=.7)
fig.text(.80,.165,'Profile split',ha='center',fontsize=10,fontweight='bold',color='#184e70')
fig.text(.80,.085,'720 train  /  100 validation\n100 test  /  80 calibration',ha='center',fontsize=9,linespacing=1.6)
for ext in ['pdf','png']:fig.savefig(OUT/f'corpus_summary.{ext}',dpi=250)
print('Corpus visual saved; every distribution sums to 1,000 profiles.')
