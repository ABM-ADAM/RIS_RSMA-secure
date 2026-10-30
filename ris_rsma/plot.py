"""Generate the eleven manuscript figures from locally computed campaign trials.
Author: Abuzar B. M. Adam
October 2026
"""
from __future__ import annotations

import csv
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from .core import Config, channel_draw, optimize_fixed

ROOT=Path.cwd()
DATA=ROOT/'outputs'/'raw';OUT=ROOT/'outputs'/'figures'
COLORS={'joint':'#0806ad','no_an':'#c00000','fixed_ris':'#287a45','no_ris':'#640068'}
NAMES={'joint':'Joint RIS/AN RSMA','no_an':'No AN','fixed_ris':'Fixed RIS','no_ris':'No RIS'}
LINESTYLES={'joint':'-','no_an':'-','fixed_ris':'--','no_ris':'-'}
MARKERS={'joint':'h','no_an':'o','fixed_ris':'^','no_ris':'o'}
plt.rcParams.update({'font.family':'serif','font.size':8.5,'axes.linewidth':.8,
 'pdf.fonttype':42,'savefig.pad_inches':.025,'legend.fontsize':7.3})


def read(family):
    path=DATA/f'{family}_raw.csv'
    if not path.exists():return []
    with path.open(newline='') as f:return list(csv.DictReader(f))


def save_table(name,headers,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(headers)
        for row in rows:w.writerow([f'{v:.10g}' if isinstance(v,(float,np.floating)) else v for v in row])


def bootstrap_mean(values,seed=20261004):
    arr=np.asarray(values,float)
    if not len(arr):
        raise ValueError('Cannot bootstrap an empty set of verified trials.')
    rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(arr,(2000,len(arr)),replace=True),axis=1)
    return float(np.mean(arr)),*map(float,np.percentile(boot,[2.5,97.5]))


def matched(family,key,methods):
    rows=read(family)
    all_x=sorted({float(r[key]) for r in rows})
    result=[]
    for x in all_x:
        source=[r for r in rows if float(r[key])==x]
        bymethod={m:{int(r['seed']):r for r in source if r['method']==m} for m in methods}
        common=set.intersection(*(set(d) for d in bymethod.values())) if all(bymethod.values()) else set()
        paired=sorted(s for s in common if all(bymethod[m][s]['status']=='verified' for m in methods))
        result.append((x,paired,bymethod,len(common)))
    return result


def plot_mean_curve(ax,method,records):
    """Reference style; bootstrap limits remain available in curve tables."""
    x,y,*_=np.asarray(records).T
    ax.plot(x,y,linestyle=LINESTYLES[method],color=COLORS[method],
            marker=MARKERS[method],markersize=3.8,markerfacecolor='white',
            markeredgecolor=COLORS[method],markeredgewidth=.85,
            linewidth=1.35,label=NAMES[method])


def fig_sweep(family,key,name,xlabel,methods):
    matches=matched(family,key,methods)
    if not matches or not any(p for _,p,_,_ in matches):
        raise ValueError(f'{name}: no paired verified trials in {family}')
    fig,ax=plt.subplots(figsize=(3.45,2.55),layout='constrained')
    files=[];availability=[];comparison=[]
    for method in methods:
        records=[]
        for x,paired,bymethod,total in matches:
            if not paired:continue
            values=[float(bymethod[method][s]['confidential']) for s in paired]
            mean,low,high=bootstrap_mean(values)
            records.append((x,mean,high-mean,mean-low,len(paired),total))
        if not records:continue
        file=f'{name}_{method}.txt';save_table(file,
            ('x','y','errplus','errminus','paired_n','total_n'),records)
        files.append((method,file,'x','y',True))
        plot_mean_curve(ax,method,records)
    if len(files)!=len(methods):
        raise ValueError(f'{name}: a comparison method has no verified trials')
    reference=methods[0]
    for x,paired,bymethod,total in matches:
        if not paired:continue
        for other in methods[1:]:
            delta=[float(bymethod[reference][s]['confidential'])
                 -float(bymethod[other][s]['confidential']) for s in paired]
            mean,low,high=bootstrap_mean(delta)
            comparison.append((x,reference,other,mean,low,high,len(paired),total))
        availability.append((x,len(paired),total,
                             *(sum(bymethod[m][s]['status']=='verified' for s in bymethod[m])
                               for m in methods)))
    save_table(f'{name}_paired_differences.txt',
        ('x','reference','other','mean_difference','ci_low','ci_high','paired_n','total_n'),comparison)
    save_table(f'{name}_availability.txt',('x','paired_n','total_n',*methods),availability)
    ax.set(xlabel=xlabel,ylabel='Confidential sum rate (bit/s/Hz)')
    ax.grid(alpha=.18)
    legend_location={'fig01_power':'upper left','fig02_uncertainty':'upper right',
                     'fig03_ris_size':'upper left'}.get(name,'best')
    ax.legend(frameon=False,loc=legend_location)
    if key in ('m','b','carrier_power_dbm'):ax.set_xticks([x for x,_,_,_ in matches])
    if name=='fig01_power':
        ax.set_yticks((10,15,20))
    elif name=='fig02_uncertainty':
        ax.set_yticks((10,15,20,25))
        ax.set_xticks(np.arange(0,.0301,.005))
        formatter=ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2,-2))
        ax.xaxis.set_major_formatter(formatter)
    fig.savefig(OUT/f'{name}.pdf');fig.savefig(OUT/f'{name}.png',dpi=360);plt.close(fig)


def fig_ecdf():
    methods=('joint','no_an','fixed_ris','no_ris')
    rows=read('blocked_main')
    if not rows:raise ValueError('fig05: missing blocked_main trials')
    by={m:{int(r['seed']):r for r in rows if r['method']==m} for m in methods}
    seeds=sorted(set.intersection(*(set(d) for d in by.values())))
    paired=[s for s in seeds if all(by[m][s]['status']=='verified' for m in methods)]
    if len(paired)<5:raise ValueError('fig05: at least five jointly verified seeds are required')
    fig,ax=plt.subplots(figsize=(3.45,2.55),layout='constrained')
    curves=[]
    for m in methods:
        values=np.sort([float(by[m][s]['confidential']) for s in paired])
        y=np.arange(1,len(values)+1)/len(values)
        file=f'fig05_{m}.txt';save_table(file,('x','y'),zip(values,y))
        curves.append((m,file,'x','y',False))
        ax.step(values,y,where='post',color=COLORS[m],label=NAMES[m],
                linestyle=LINESTYLES[m],lw=1.3)
    comparisons=[]
    for other in methods[1:]:
        delta=[float(by['joint'][s]['confidential'])-float(by[other][s]['confidential']) for s in paired]
        mean,low,high=bootstrap_mean(delta)
        comparisons.append((other,mean,low,high,len(paired),len(seeds),sum(v>0 for v in delta)))
    save_table('fig05_paired_differences.txt',
        ('other','mean_difference','ci_low','ci_high','paired_n','total_n','wins'),comparisons)
    ax.set(xlabel='Confidential sum rate (bit/s/Hz)',ylabel='Empirical CDF',ylim=(0,1.01))
    ax.grid(alpha=.18);ax.legend(frameon=False,loc='lower right')
    fig.savefig(OUT/'fig05_confidential_rate_ecdf.pdf')
    fig.savefig(OUT/'fig05_confidential_rate_ecdf.png',dpi=360);plt.close(fig)


def fig_tradeoff():
    methods=('joint','no_ris')
    matches=matched('blocked_tradeoff','lambda_public',methods)
    if not matches or not any(p for _,p,_,_ in matches):
        raise ValueError('fig04: no paired verified tradeoff trials')
    fig,ax=plt.subplots(figsize=(3.45,2.55),layout='constrained');curves=[];deltas=[]
    for method in methods:
        records=[]
        for lam,paired,by,total in matches:
            if not paired:continue
            pub=np.mean([float(by[method][s]['public']) for s in paired])
            sec,low,high=bootstrap_mean([float(by[method][s]['confidential']) for s in paired])
            records.append((pub,sec,high-sec,sec-low,lam,len(paired),total))
        if not records:continue
        name=f'fig04_{method}.txt';save_table(name,
            ('x','y','errplus','errminus','lambda','paired_n','total_n'),records)
        curves.append((method,name,'x','y',True))
        plot_mean_curve(ax,method,records)
    for lam,paired,by,total in matches:
        if not paired:continue
        diffs=[float(by['joint'][s]['objective'])-float(by['no_ris'][s]['objective']) for s in paired]
        mean,lo,hi=bootstrap_mean(diffs)
        deltas.append((lam,mean,lo,hi,len(paired),total))
    save_table('fig04_paired_objective_differences.txt',
        ('lambda','mean_difference','ci_low','ci_high','paired_n','total_n'),deltas)
    ax.set(xlabel='Public sum rate (bit/s/Hz)',ylabel='Confidential sum rate (bit/s/Hz)')
    ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.savefig(OUT/'fig04_public_confidential_tradeoff.pdf')
    fig.savefig(OUT/'fig04_public_confidential_tradeoff.png',dpi=360);plt.close(fig)


def fig_an():
    rows=read('blocked_main')
    valid=[r for r in rows if r['method']=='joint' and r['status']=='verified']
    if not valid:raise ValueError('fig06: no verified joint-design trials')
    x=(0.,.05,.1,.2,.3);counts=[sum(abs(float(r['an_total_power_fraction'])-q)<1e-8 for r in valid) for q in x]
    save_table('fig06_an_selection.txt',('an_power_fraction','count','verified_n','total_n'),
        [(q,c,len(valid),sum(r['method']=='joint' for r in rows)) for q,c in zip(x,counts)])
    fig,ax=plt.subplots(figsize=(3.45,2.55),layout='constrained')
    ax.bar(np.arange(len(x)),np.array(counts)/len(valid),color=COLORS['joint'],width=.65)
    ax.set(xticks=np.arange(len(x)),xticklabels=[f'{q:g}' for q in x],ylim=(0,1),
           xlabel='Total AN power fraction $Nq/P_B$',ylabel='Selection frequency')
    ax.grid(axis='y',alpha=.18)
    fig.savefig(OUT/'fig06_an_selection.pdf');fig.savefig(OUT/'fig06_an_selection.png',dpi=360)
    plt.close(fig)


def fig_convergence():
    rows=read('blocked_main')
    valid=sorted((r for r in rows if r['method']=='joint' and r['status']=='verified'),
                 key=lambda r:float(r['confidential']))
    if not valid:raise ValueError('fig07: no verified joint-design trials')
    # Median rule is declared before trace computation; fixed RIS and AN pair.
    seed=int(valid[len(valid)//2]['seed'])
    cfg=replace(Config(),scenario='blocked',m=256);draw=channel_draw(seed,cfg)
    fit=optimize_fixed(draw,cfg,np.ones(cfg.m,complex),q_fraction=.05,
                       phase_budget=12,inner_budget=20)
    if fit['status']!='verified':
        raise ValueError(f'fig07: fixed-pair diagnostic failed for seed {seed}')
    trace=fit['trace'];rows=[(i,float(v),seed,.05) for i,v in enumerate(trace)]
    save_table('fig07_fixed_pair_convergence.txt',
               ('iteration','objective','seed','an_power_fraction'),rows)
    fig,ax=plt.subplots(figsize=(3.45,2.55),layout='constrained')
    ax.plot(np.arange(len(trace)),trace,color=COLORS['joint'],lw=1.35,
            marker=MARKERS['joint'],markersize=3.8,markerfacecolor='white',
            markeredgecolor=COLORS['joint'],markeredgewidth=.85)
    ax.set(xlabel='Iteration',ylabel='Unpenalized objective (bit/s/Hz)',
           xlim=(0,max(20,len(trace)-1)))
    ax.set_ylim(np.floor(min(trace))-1,np.ceil(max(trace)))
    if 10 <= min(trace) and max(trace) <= 15:
        ax.set_yticks((10,12,14))
    ax.grid(alpha=.18)
    fig.savefig(OUT/'fig07_fixed_pair_convergence.pdf')
    fig.savefig(OUT/'fig07_fixed_pair_convergence.png',dpi=360);plt.close(fig)


def fig_scenario_gain():
    records=[]
    for x,family in ((1,'umi_main'),(2,'blocked_main')):
        rows=read(family)
        by={m:{int(r['seed']):r for r in rows if r['method']==m}
            for m in ('joint','no_ris')}
        common=sorted(set(by['joint'])&set(by['no_ris']))
        paired=[s for s in common if all(by[m][s]['status']=='verified'
                                         for m in by)]
        if not paired:continue
        delta=[float(by['joint'][s]['confidential'])-
               float(by['no_ris'][s]['confidential']) for s in paired]
        mean,low,high=bootstrap_mean(delta)
        records.append((x,mean,high-mean,mean-low,len(paired),len(common)))
    if len(records)!=2:raise ValueError('fig08: both scenarios require paired verified trials')
    save_table('fig08_scenario_gain.txt',
       ('scenario','gain','errplus','errminus','paired_n','total_n'),records)
    fig,ax=plt.subplots(figsize=(3.45,2.55),layout='constrained')
    x,y,ep,em,*_=np.asarray(records).T
    ax.bar(x,y,width=.42,facecolor=COLORS['joint'],alpha=.18,
           edgecolor=COLORS['joint'],linewidth=1.1)
    ax.vlines(x,y-em,y+ep,color=COLORS['joint'],linewidth=.8)
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xticks=x,xticklabels=['Random LOS','Blocked corridor'],xlim=(.5,2.5),
           ylabel='RIS gain (bit/s/Hz)')
    ax.grid(axis='y',alpha=.18)
    fig.savefig(OUT/'fig08_scenario_gain.pdf')
    fig.savefig(OUT/'fig08_scenario_gain.png',dpi=360);plt.close(fig)


def fig_runtime():
    methods=('joint','fixed_ris','no_ris')
    matches=matched('blocked_ris_size','m',methods)
    if not matches or not any(p for _,p,_,_ in matches):
        raise ValueError('fig10: no paired verified RIS-size trials')
    fig,ax=plt.subplots(figsize=(3.45,2.55),layout='constrained')
    curves=[]
    for method in methods:
        records=[]
        for m,paired,by,total in matches:
            if not paired:continue
            values=[float(by[method][s]['runtime_s']) for s in paired]
            mean,low,high=bootstrap_mean(values)
            records.append((m,mean,high-mean,mean-low,len(paired),total))
        if not records:continue
        filename=f'fig10_runtime_{method}.txt'
        save_table(filename,('x','y','errplus','errminus','paired_n','total_n'),records)
        curves.append((method,filename,'x','y',True))
        plot_mean_curve(ax,method,records)
    if len(curves)!=len(methods):
        raise ValueError('fig10: a comparison method has no verified trials')
    ax.set(xlabel='RIS elements $M$',ylabel='Measured search time (s)')
    ax.set_xticks([m for m,_,_,_ in matches]);ax.grid(alpha=.18)
    ax.legend(frameon=False)
    fig.savefig(OUT/'fig10_runtime.pdf');fig.savefig(OUT/'fig10_runtime.png',dpi=360)
    plt.close(fig)


def fig_threshold_goodput():
    methods=('joint','no_ris')
    rows=read('blocked_threshold')
    if not rows:raise ValueError('fig09: missing threshold trials')
    fig,ax=plt.subplots(figsize=(3.45,2.55),layout='constrained')
    curves=[];availability=[];diff=[]
    for method in methods:
        records=[]
        for floor in sorted({float(r['secret_floor']) for r in rows}):
            by={int(r['seed']):r for r in rows
                if float(r['secret_floor'])==floor and r['method']==method}
            if not by:continue
            values=[float(r['confidential']) if r['status']=='verified' else 0.
                    for r in by.values()]
            mean,low,high=bootstrap_mean(values)
            records.append((floor,mean,high-mean,mean-low,
                            sum(r['status']=='verified' for r in by.values()),len(by)))
        filename=f'fig09_rate_requirement_{method}.txt'
        save_table(filename,('x','y','errplus','errminus','verified_n','total_n'),records)
        curves.append((method,filename,'x','y',True))
        plot_mean_curve(ax,method,records)
    for floor in sorted({float(r['secret_floor']) for r in rows}):
        selected=[r for r in rows if float(r['secret_floor'])==floor]
        by={m:{int(r['seed']):r for r in selected if r['method']==m} for m in methods}
        common=sorted(set(by['joint'])&set(by['no_ris']))
        if not common:continue
        values=lambda m,s:(float(by[m][s]['confidential'])
                            if by[m][s]['status']=='verified' else 0.)
        delta=[values('joint',s)-values('no_ris',s) for s in common]
        mean,low,high=bootstrap_mean(delta)
        diff.append((floor,mean,low,high,len(common)))
        availability.append((floor,len(common),
             sum(by['joint'][s]['status']=='verified' for s in common),
             sum(by['no_ris'][s]['status']=='verified' for s in common)))
    save_table('fig09_rate_requirement_paired_differences.txt',
       ('secret_floor','mean_gain','ci_low','ci_high','paired_n'),diff)
    save_table('fig09_rate_requirement_availability.txt',
       ('secret_floor','paired_n','joint_verified','no_ris_verified'),availability)
    ax.set(xlabel='Minimum per-user confidential rate (bit/s/Hz)',
           ylabel='Verified confidential goodput (bit/s/Hz)')
    ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.savefig(OUT/'fig09_rate_requirement.pdf')
    fig.savefig(OUT/'fig09_rate_requirement.png',dpi=360);plt.close(fig)


def _figure_specs():
    """Figure names, required campaign families, and plot functions."""
    return {
        'fig01_power': (('blocked_power',), lambda: fig_sweep(
            'blocked_power','carrier_power_dbm','fig01_power',
            'Carrier BS power (dBm)',('joint','no_ris'))),
        'fig02_uncertainty': (('blocked_uncertainty',), lambda: fig_sweep(
            'blocked_uncertainty','uncertainty_relative','fig02_uncertainty',
            'Primitive CSI error fraction',('joint','no_an'))),
        'fig03_ris_size': (('blocked_ris_size',), lambda: fig_sweep(
            'blocked_ris_size','m','fig03_ris_size','RIS elements $M$',
            ('joint','fixed_ris','no_ris'))),
        'fig04_public_confidential_tradeoff': (('blocked_tradeoff',),fig_tradeoff),
        'fig05_confidential_rate_ecdf': (('blocked_main',),fig_ecdf),
        'fig06_an_selection': (('blocked_main',),fig_an),
        'fig07_fixed_pair_convergence': (('blocked_main',),fig_convergence),
        'fig08_scenario_gain': (('umi_main','blocked_main'),fig_scenario_gain),
        'fig09_rate_requirement': (('blocked_threshold',),fig_threshold_goodput),
        'fig10_runtime': (('blocked_ris_size',),fig_runtime),
        'fig11_phase_resolution': (('blocked_phase_bits',), lambda: fig_sweep(
            'blocked_phase_bits','b','fig11_phase_resolution',
            'RIS phase resolution $B$ (bits)',('joint','fixed_ris'))),
    }


CURVE_TABLES={
    'fig01_power':('fig01_power_joint','fig01_power_no_ris'),
    'fig02_uncertainty':('fig02_uncertainty_joint','fig02_uncertainty_no_an'),
    'fig03_ris_size':('fig03_ris_size_joint','fig03_ris_size_fixed_ris','fig03_ris_size_no_ris'),
    'fig04_public_confidential_tradeoff':('fig04_joint','fig04_no_ris'),
    'fig05_confidential_rate_ecdf':tuple(f'fig05_{m}' for m in ('joint','no_an','fixed_ris','no_ris')),
    'fig06_an_selection':('fig06_an_selection',),
    'fig07_fixed_pair_convergence':('fig07_fixed_pair_convergence',),
    'fig08_scenario_gain':('fig08_scenario_gain',),
    'fig09_rate_requirement':('fig09_rate_requirement_joint','fig09_rate_requirement_no_ris'),
    'fig10_runtime':('fig10_runtime_joint','fig10_runtime_fixed_ris','fig10_runtime_no_ris'),
    'fig11_phase_resolution':('fig11_phase_resolution_joint','fig11_phase_resolution_fixed_ris'),
}


def consolidate_curve_values(name):
    """Write a single long-form numeric table for a plotted figure."""
    records=[]
    for source in CURVE_TABLES[name]:
        with (OUT/f'{source}.txt').open(newline='') as f:
            data=list(csv.DictReader(f,delimiter='\t'))
        for row in data:
            if 'x' in row:x,y=float(row['x']),float(row['y'])
            elif 'iteration' in row:x,y=float(row['iteration']),float(row['objective'])
            elif 'an_power_fraction' in row:
                x=float(row['an_power_fraction'])
                y=float(row['count'])/float(row['verified_n'])
            else:x,y=float(row['scenario']),float(row['gain'])
            ep,em=float(row.get('errplus',0)),float(row.get('errminus',0))
            n=(1 if 'iteration' in row else
               int(row.get('paired_n',row.get('verified_n',len(data)))))
            total=int(row.get('total_n',n))
            records.append((source,x,y,ep,em,y-em,y+ep,n,total,
                            row.get('lambda','')))
    save_table(name+'_values.txt',
        ('curve','x','y','errplus','errminus','y_low','y_high',
         'n','total_n','lambda_public'),records)


def main(argv=None):
    global DATA,OUT
    specs=_figure_specs()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--figure',choices=('all',*specs),default='all')
    args=parser.parse_args(argv)
    DATA=args.data_dir.resolve();OUT=args.output_dir.resolve()
    selected=list(specs) if args.figure=='all' else [args.figure]
    required={family for name in selected for family in specs[name][0]}
    missing=[f'{family}_raw.csv' for family in sorted(required)
             if not (DATA/f'{family}_raw.csv').is_file()]
    if missing:
        parser.error('Run the required campaign families first: '+', '.join(missing))
    OUT.mkdir(parents=True,exist_ok=True)
    for name in selected:
        specs[name][1]()
        consolidate_curve_values(name)
        print(f'Generated {name}',flush=True)


if __name__=='__main__':main()
