"""Regenerate the article's figures from the completed experiment summaries."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CITIES = {'tel_aviv': 'Tel Aviv–Gush Dan', 'amsterdam': 'Amsterdam', 'nyc': 'New York City area'}
METHODS = {'ridge': ('Ridge', '#77808c'), 'boosted_trees': ('Boosted trees', '#146b8a'),
           'mlp': ('MLP', '#d17b29'), 'dp': ('Landmark DP', '#8c4f7d')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('article/figures'))
    args = parser.parse_args()
    data = json.loads((args.results / 'audits.json').read_text())
    audits = data['audits']
    expected = {(city, res) for city in CITIES for res in (6, 7, 8)}
    assert data['completion']['complete'] and len(audits) == 9
    assert {(a['city'], a['resolution']) for a in audits} == expected
    assert all({m['method'] for m in a['metrics']} == set(METHODS) for a in audits)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelcolor': '#333333',
                         'text.color': '#252525', 'axes.titleweight': 'bold', 'svg.fonttype': 'path', 'svg.hashsalt': 'duration-approximation-v6'})
    for city, label in CITIES.items():
        group = sorted((a for a in audits if a['city'] == city), key=lambda a: a['resolution'])
        fig, ax = plt.subplots(figsize=(7, 4.4), layout='constrained')
        for method, (name, color) in METHODS.items():
            ys = [next(m['mae_seconds'] for m in a['metrics'] if m['method'] == method) for a in group]
            ax.plot([6, 7, 8], ys, marker='o', color=color, label=name, linewidth=2)
        ax.set(title=label, xlabel='H3 resolution → smaller cells', ylabel='Mean absolute error (seconds)',
               xticks=[6, 7, 8], ylim=(0, 600))
        ax.grid(axis='y', color='#eeeeee'); ax.legend(ncol=2, frameon=False, loc='upper right', fontsize=11.5)
        fig.savefig(args.output / f'accuracy-{city}.svg', metadata={'Date': None}); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.6), layout='constrained')
    for (city, label), color, marker in zip(CITIES.items(), ['#146b8a','#d17b29','#8c4f7d'], ['o','s','^']):
        group = sorted((a for a in audits if a['city'] == city), key=lambda a: a['resolution'])
        xs = [a['precomputation']['matrix_payload_bytes'] / 1_000_000 for a in group]
        ys = [next(m['mae_seconds'] for m in a['metrics'] if m['method'] == 'boosted_trees') for a in group]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2, label=label)
        for x, y, a in zip(xs, ys, group):
            ax.annotate(f"r{a['resolution']}", (x, y), xytext=(5, -14 if city == 'amsterdam' else 8), textcoords='offset points', fontsize=10)
    ax.set(xscale='log', xlabel='Directed landmark matrix (MB, logarithmic scale)',
           ylabel='Boosted-tree MAE (seconds)', ylim=(60, 145), title='More precomputation buys lower error')
    ax.grid(color='#eeeeee'); ax.legend(frameon=False, loc='upper right', fontsize=11.5)
    fig.savefig(args.output / 'accuracy-storage.svg', metadata={'Date': None}); plt.close(fig)
    # A conceptual illustration: positions and arrows are not measured routes.
    fig, ax = plt.subplots(figsize=(7, 3.8), layout='constrained')
    ax.set(xlim=(-1.6, 6.6), ylim=(-1.8, 1.6), aspect='equal'); ax.axis('off')
    for center, title in [(0, 'Pickup cell'), (5, 'Drop-off cell')]:
        angles=np.arange(6)*np.pi/3
        vertices=np.column_stack([center+np.cos(angles), np.sin(angles)])
        ax.fill(vertices[:,0], vertices[:,1], color='#f0f5f6', edgecolor='#8da4ad', linewidth=1.5)
        ax.scatter(vertices[:,0],vertices[:,1],color='#146b8a',s=35,zorder=4)
        ax.text(center,1.3,title,ha='center',weight='bold')
    p=(-.25,-.25); li=(1,0); lj=(4,0); q=(5.3,.2)
    ax.scatter(*p,color='#252525',s=50,zorder=5);ax.scatter(*q,color='#252525',s=50,zorder=5)
    ax.text(p[0]-.25,p[1]-.38,'p',fontsize=13);ax.text(q[0]+.12,q[1]+.1,'q',fontsize=13)
    for a,b,color in [(p,li,'#d17b29'),(li,lj,'#146b8a'),(lj,q,'#d17b29')]:
        ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':color,'lw':2.5,'shrinkA':5,'shrinkB':5})
    ax.text(2.5,.24,'Precomputed duration',ha='center',color='#146b8a',fontsize=11)
    ax.text(1,-.38,'ℓᵢ',ha='center',fontsize=13);ax.text(4,-.38,'ℓⱼ',ha='center',fontsize=13)
    ax.text(2.5,-1.5,'Choose the cheapest legal combination of three legs',ha='center',fontsize=11)
    fig.savefig(args.output / 'landmark-route.svg', metadata={'Date': None});plt.close(fig)
    for path in args.output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    print(f'Wrote 5 figures to {args.output}')


if __name__ == '__main__':
    main()
