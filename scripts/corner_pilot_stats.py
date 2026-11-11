"""Descriptive numbers for the #169 test-server pilot files (docs/corner-labels-169-pilot.md).

Reads the split na_discount_sw proposals, the pilot order and the split precision sample;
writes pilot_residential/pilot_stats.json and the pano-id lists panos.<file>.txt (the input
of the read-only pano-presence query on the clearance sheet). Stdlib only. Run from the repo
root: python scripts/corner_pilot_stats.py
"""
import json, csv, statistics, collections, sys
D='runs/vancouver/corner_labels169_split/na_discount_sw'
P=D+'/pilot_residential'
props={}
for l in open(D+'/proposals.jsonl',encoding='utf-8'):
    p=json.loads(l); props[p['key']]=p
order=[f"{r['unit']}#{r['corner']}" for r in csv.DictReader(open(P+'/pilot.csv',encoding='utf-8'))]
samp=[f"{r['unit']}#{r['corner']}" for r in csv.DictReader(open(D+'/sample.csv',encoding='utf-8'))]
rated_res=[k for k in samp if props[k]['stratum']=='residential']
def q(xs,f):
    xs=sorted(xs); i=f*(len(xs)-1); lo=int(i); hi=min(lo+1,len(xs)-1); return xs[lo]+(xs[hi]-xs[lo])*(i-lo)
out={}
for name,keys in (('pilot200',order[:200]),('rated100+20',order[:120]),('residential',order)):
    ps=[props[k] for k in keys]
    panos=collections.Counter(p['pano']['pano_id'] for p in ps)
    r=[p['range_m'] for p in ps]
    ph=sum(1 for p in ps if p['source']['sector'].get('phantom_start') or p['source']['sector'].get('phantom_end'))
    wide=sum(1 for p in ps if p['source']['sector'].get('wide'))
    runs=collections.Counter(p['pano']['run'] for p in ps)
    inv=collections.Counter(p['source']['inventory']['status'] for p in ps)
    yrs=collections.Counter((p['pano'].get('capture_date') or '????')[:4] for p in ps)
    wh=collections.Counter((p['pano']['width'],p['pano']['height']) for p in ps)
    out[name]=dict(labels=len(ps),panos=len(panos),labels_per_pano=round(len(ps)/len(panos),3),
        max_per_pano=max(panos.values()),range_p10_med_p90=[round(q(r,.1),1),round(statistics.median(r),1),round(q(r,.9),1)],
        phantom=ph,phantom_share=round(ph/len(ps),3),wide=wide,wide_share=round(wide/len(ps),3),
        runs=dict(runs),inventory=dict(inv),years=dict(sorted(yrs.items())),dims=dict((f'{a}x{b}',n) for (a,b),n in wh.items()),
        rated_in=sum(1 for k in keys if k in set(rated_res)))

# collisions: panos of the 200 that hold other residential corners outside the 200
p200=set(props[k]['pano']['pano_id'] for k in order[:200])
other=[k for k in order[200:] if props[k]['pano']['pano_id'] in p200]
out['complement_on_pilot_panos']=dict(corners=len(other),panos=len({props[k]['pano']['pano_id'] for k in other}))
# position of jon's rated corners in the order
pos={k:i for i,k in enumerate(order,1)}
out['rated_positions_le200']=sorted(pos[k] for k in rated_res if pos[k]<=200)
out['split_sample_residential_rated'] = len(rated_res)
with open(P + '/pilot_stats.json', 'w', newline='\n') as f:
    f.write(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
# pano lists for DB
for name,keys in (('pilot200',order[:200]),('residential',order)):
    with open(P+f'/panos.{name}.txt','w',newline='\n') as f:
        for pid in sorted({props[k]['pano']['pano_id'] for k in keys}): f.write(pid+'\n')
