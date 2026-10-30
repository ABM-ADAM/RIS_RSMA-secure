"""Run paired, documented channel experiments. See README.md before use.

The output CSV is append-only within a run and includes every trial, including
failed feasible-start searches. A failure is not a mathematical infeasibility
certificate. The row data, not a plotting script, is the source of figures.

Author: Abuzar B. M. Adam
October 2026
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from .core import Config, aperture_subset, channel_draw, screened_search


FAMILIES={
 'blocked_main':[(30.,.01,256,.2,m) for m in ('joint','no_an','fixed_ris','no_ris')],
 'blocked_power':[(p,.01,256,.2,m) for p in (20.,25.,30.,35.) for m in ('joint','no_ris')],
 'blocked_uncertainty':[(30.,e,256,.2,m) for e in (0.,.005,.01,.02,.03) for m in ('joint','no_an')],
 'blocked_ris_size':[(30.,.01,m,.2,method) for m in (64,144,256) for method in ('joint','fixed_ris','no_ris')],
 'blocked_tradeoff':[(30.,.01,256,lam,m) for lam in (0.,.1,.2,.5,1.) for m in ('joint','no_ris')],
 'blocked_threshold':[(30.,.01,256,.2,m,r) for r in (0.,.25,.5,1.,1.5) for m in ('joint','no_ris')],
 'blocked_phase_bits':[(30.,.01,256,.2,m,0.,b) for b in (1,2,3,4)
                       for m in ('joint','fixed_ris')],
 'umi_main':[(30.,.01,256,.2,m) for m in ('joint','no_an','fixed_ris','no_ris')],
}
PAPER_DRAWS={'blocked_main':100,'umi_main':100,'blocked_power':50,
             'blocked_uncertainty':40,'blocked_ris_size':40,
             'blocked_tradeoff':40,'blocked_threshold':40,
             'blocked_phase_bits':40}
PROJECT_ROOT=Path.cwd()
FIELDNAMES=(
    'seed','family','scenario','n','carrier_power_dbm','prb_power_dbm',
    'uncertainty_relative','m','b','lambda_public','secret_floor','method',
    'status','candidates','candidate_feasible','solver_calls','runtime_s',
    'objective','confidential','public','an_total_power_fraction',
    'minimum_private_secrecy','minimum_constraint_residual',
)


def job(task):
    seed,family,condition,candidate_budget,phase_budget,inner_budget=task
    power,epsilon,m,lam,mode=condition[:5]
    floor=condition[5] if len(condition)>5 else 0.
    phase_bits=condition[6] if len(condition)>6 else 2
    cfg=replace(Config(),carrier_power_dbm=power,epsilon_relative=epsilon,
                m=m,b=phase_bits,lambda_public=lam,secret_floor=floor,
                scenario='blocked' if family.startswith('blocked_')
                else 'unconditioned_corridor')
    if family=='blocked_ris_size':
        full=channel_draw(seed,replace(cfg,m=256))
        draw=aperture_subset(full,256,m,epsilon)
    else:draw=channel_draw(seed,cfg)
    out=screened_search(draw,cfg,seed,mode,candidate_budget,phase_budget,inner_budget)
    row=dict(seed=seed,family=family,scenario=cfg.scenario,n=cfg.n,
             carrier_power_dbm=power,
             prb_power_dbm=power+10*np.log10(cfg.prb_bandwidth_hz/cfg.carrier_bandwidth_hz),
             uncertainty_relative=epsilon,m=m,b=cfg.b,lambda_public=lam,
             secret_floor=floor,
             method=mode,status=out['status'],candidates=out['tested'],
             candidate_feasible=out['feasible'],solver_calls=out['solves'],
             runtime_s=out['seconds'],objective='',confidential='',public='',
             an_total_power_fraction='',minimum_private_secrecy='',
             minimum_constraint_residual='')
    if out['status']=='verified':
        row.update(objective=out['objective'],confidential=out['secret'],
                   public=out['public'],an_total_power_fraction=out['q_fraction'],
                   minimum_private_secrecy=out['min_secret'],
                   minimum_constraint_residual=out['residual'])
    return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=tuple(FAMILIES)+('all',),default='blocked_main')
    p.add_argument('--paper',action='store_true',help='Use all eight families and the archived per-family draw counts.')
    p.add_argument('--draws',type=int,default=40,help='Draws per operating point unless --paper is set.')
    p.add_argument('--first-seed',type=int,default=1000)
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--candidates',type=int,default=12)
    p.add_argument('--phase-iterations',type=int,default=6)
    p.add_argument('--inner-iterations',type=int,default=4)
    p.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs'/'raw')
    p.add_argument('--resume',action='store_true',
                   help='Append only missing tasks if the run metadata matches.')
    p.add_argument('--dry-run',action='store_true',
                   help='Show planned trials without creating any files.')
    args=p.parse_args(argv)
    if min(args.draws,args.workers,args.candidates)<1 or min(args.phase_iterations,args.inner_iterations)<0:
        p.error('draws, workers and candidates must be positive; iteration budgets must be nonnegative')
    families=FAMILIES if args.paper or args.family=='all' else {args.family:FAMILIES[args.family]}
    if args.dry_run:
        total=0
        for family,conditions in families.items():
            count=PAPER_DRAWS[family] if args.paper else args.draws
            trials=count*len(conditions)
            print(f'{family}: {count} seeds x {len(conditions)} settings = {trials} trials')
            total+=trials
        print(f'Total planned trials: {total}')
        return
    args.out.mkdir(parents=True,exist_ok=True)
    for family,conditions in families.items():
        path=args.out/f'{family}_raw.csv'
        metadata_path=args.out/f'{family}_run.json'
        metadata=dict(schema_version=1,family=family,conditions=[list(c) for c in conditions],
            candidates=args.candidates,phase_iterations=args.phase_iterations,
            inner_iterations=args.inner_iterations,first_seed=args.first_seed)
        new_file=not path.exists()
        if path.exists() and not args.resume:
            raise FileExistsError(f'Existing run: {path}; use --resume if appropriate')
        if path.exists():
            if not metadata_path.exists() or json.loads(metadata_path.read_text())!=metadata:
                raise ValueError(f'Cannot resume {path}: missing or mismatched run settings')
        elif metadata_path.exists():
            raise FileExistsError(f'Run metadata exists without CSV: {metadata_path}')
        else:
            metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
        count=PAPER_DRAWS[family] if args.paper else args.draws
        tasks=[(seed,family,cond,args.candidates,args.phase_iterations,args.inner_iterations)
               for seed in range(args.first_seed,args.first_seed+count) for cond in conditions]
        existing=set()
        if path.exists():
            with path.open(newline='') as f:
                reader=csv.DictReader(f)
                if reader.fieldnames!=list(FIELDNAMES):
                    raise ValueError(f'Incomplete or changed CSV schema in {path}')
                for r in reader:
                    existing.add((int(r['seed']),float(r['carrier_power_dbm']),
                        float(r['uncertainty_relative']),int(r['m']),
                        float(r['lambda_public']),r['method'],float(r['secret_floor']),int(r['b'])))
            tasks=[t for t in tasks if (t[0],float(t[2][0]),float(t[2][1]),
                    int(t[2][2]),float(t[2][3]),t[2][4],
                    float(t[2][5] if len(t[2])>5 else 0.),
                    int(t[2][6] if len(t[2])>6 else 2)) not in existing]
        with ProcessPoolExecutor(max_workers=args.workers) as executor,open(path,'a' if path.exists() else 'w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDNAMES)
            if new_file:writer.writeheader()
            futures=[executor.submit(job,task) for task in tasks]
            for i,future in enumerate(as_completed(futures),1):
                row=future.result()
                writer.writerow(row);stream.flush()
                if i%max(1,len(conditions)*5)==0:
                    print(f'{family}: {i}/{len(tasks)} evaluated',flush=True)
        print(f'completed {family}: {path}',flush=True)


if __name__=='__main__':main()
