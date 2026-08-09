import sys
sys.path.insert(0, '.')
from src import config as cfgmod
base = cfgmod.load_base_config('configs/base_config.yaml')
exps = cfgmod.load_experiments('configs/experiments.yaml')
print('Base config OK:', list(base.keys()))
for e in exps:
    resolved = cfgmod.resolve_experiment(base, e)
    name = resolved['name']
    mname = resolved['model']['name']
    lname = resolved['loss']['name']
    nch = len(resolved['channels'])
    print(f'  {name}: {mname} / {lname} / {nch}ch')
print('All experiments valid.')
