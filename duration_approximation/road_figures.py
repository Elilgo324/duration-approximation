"""Draw road-pattern illustrations from the bundled, attributed OSM geometry.

These cartographic extracts are separate from the experiment's frozen OSRM maps.
No routing, resampling, training, or metric calculation takes place here.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np

CENTERS = {'tel_aviv': (34.81, 32.08), 'amsterdam': (4.9, 52.365), 'nyc': (-73.975, 40.75)}
LABELS = {'tel_aviv': 'Tel Aviv–Gush Dan', 'amsterdam': 'Amsterdam', 'nyc': 'New York City area'}
WIDTH_KM = 12
MAJOR = {'motorway', 'trunk', 'primary', 'secondary'}


def xy(coordinates, center):
    """Local equirectangular coordinates in km, with north up."""
    return (np.asarray(coordinates) - center) * [111.195 * np.cos(np.deg2rad(center[1])), 111.195]


def scale_bar(ax, extent, length):
    xmin, xmax, ymin, ymax = extent
    x, y = xmin + .065*(xmax-xmin), ymin + .065*(ymax-ymin)
    ax.plot([x, x+length], [y,y], color='#263b46', lw=2.2, zorder=8)
    ax.plot([x,x], [y-.07*length,y+.07*length], color='#263b46', lw=1.2, zorder=8)
    ax.plot([x+length,x+length], [y-.07*length,y+.07*length], color='#263b46', lw=1.2, zorder=8)
    ax.text(x+length/2,y+.14*length,f'{length:g} km',ha='center',va='bottom',fontsize=11,
            bbox={'facecolor':'#faf9f5','edgecolor':'none','pad':2},zorder=9)
    ax.annotate('N', xy=(.94,.91), xytext=(.94,.82), xycoords='axes fraction', ha='center',
                fontsize=11, color='#263b46', arrowprops={'arrowstyle':'-|>','color':'#263b46','lw':1})


def render(ax, roads, center, extent, overview=False):
    groups={False: [], True: []}
    for road in roads:
        kind=road.get('tags',{}).get('highway','')
        # Omit service/parking access roads so the street structure remains legible.
        if kind == 'service': continue
        geometry=road.get('geometry',[])
        if len(geometry)<2:continue
        line=xy([[p['lon'],p['lat']] for p in geometry],center)
        xmin,xmax,ymin,ymax=extent
        if line[:,0].max()<xmin or line[:,0].min()>xmax or line[:,1].max()<ymin or line[:,1].min()>ymax:continue
        groups[kind.removesuffix('_link') in MAJOR].append(line)
    for major in (False,True):
        ax.add_collection(LineCollection(groups[major],colors='#146b8a' if major else '#65717b',
            linewidths=(.45 if major else .17) if overview else (.85 if major else .36),
            alpha=.92 if major else .78,rasterized=True))
    ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xticks=[],yticks=[])
    ax.set_facecolor('#faf9f5')
    for spine in ax.spines.values():spine.set_color('#e1e5e6');spine.set_linewidth(.7)
    scale_bar(ax,extent,5 if overview else 2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('article/map-data'))
    parser.add_argument('--output',type=Path,default=Path('article/figures'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#263b46'})
    manifest={'purpose':'Cartographic context only; not a new experiment',
              'projection':'Local equirectangular; x uses cosine of each city center latitude',
              'window_width_km':WIDTH_KM,'north_up':True,'service_roads_displayed':False,
              'road_colors':{'major':'#146b8a','other':'#65717b'},'cities':{}}
    composite, axes=plt.subplots(1,3,figsize=(13.8,5),layout='constrained')
    for city,center in CENTERS.items():
        identity=json.loads((args.data/f'{city}_identity.json').read_text())
        raw=gzip.decompress((args.data/f'{city}.json.gz').read_bytes())
        assert hashlib.sha256(raw).hexdigest()==identity['geometry_sha256'],city
        roads=json.loads(raw)['elements'];extent=(-6,6,-6,6)
        for overview in (False,True):
            if overview:
                w,s,e,n=identity['bbox_wsen'];corners=xy([[w,s],[e,n]],center)
                current=(corners[0,0],corners[1,0],corners[0,1],corners[1,1])
                xspan=current[1]-current[0];yspan=current[3]-current[2]
                size=(6,6*yspan/xspan)
            else: current=extent;size=(6,6)
            fig,ax=plt.subplots(figsize=size)
            fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
            render(ax,roads,center,current,overview)
            if overview:
                ax.add_patch(Rectangle((-6,-6),12,12,fill=False,edgecolor='#c06f35',linewidth=1.4,zorder=6))
            suffix='study-region' if overview else 'streets'
            fig.savefig(args.output/f'{city}-{suffix}.png',dpi=200,facecolor='white',
                        metadata={'Description':f'{LABELS[city]}; OSM roads; © OpenStreetMap contributors (ODbL). See article/map-data for provenance.'})
            plt.close(fig)
        ax=axes[list(CENTERS).index(city)];render(ax,roads,center,extent)
        ax.set_title(LABELS[city],fontsize=15,pad=12)
        manifest['cities'][city]={'center_lon_lat':center,'study_bbox_wsen':identity['bbox_wsen'],
                                 'data_sha256':identity['geometry_sha256']}
    composite.suptitle('Three street patterns, the same 12 × 12 km window',fontsize=18,weight='bold')
    composite.supxlabel('Major roads in blue · Other streets in gray · North up · © OpenStreetMap contributors',fontsize=11)
    composite.savefig(args.output/'city-road-comparison.png',dpi=180,facecolor='white');plt.close(composite)
    (args.data/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved six road maps, comparison figure, and map manifest')


if __name__=='__main__':
    main()
