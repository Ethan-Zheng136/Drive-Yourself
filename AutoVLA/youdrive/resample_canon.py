"""Resample GTRS/Hydra dumps (40pts/0.1s) -> 8pts/0.5s to match the other models."""
import json, numpy as np
CMP="/mnt/pfs/zhengguantian/autovla/compare_full"
def resample(path_in, path_out, name):
    d=json.load(open(path_in)); meta=d["_meta"]
    H=float(meta["horizon_s"]); n=int(meta["n_pts"])  # 4.0, 40
    t_src=np.linspace(H/n, H, n)                       # 0.1,0.2,...,4.0
    t_dst=np.linspace(0.5, 4.0, 8)                     # 8 pts @0.5s
    out={"_meta":{"model":name,"horizon_s":4.0,"n_pts":8,"frame":meta["frame"]}}
    for k,v in d.items():
        if k=="_meta": continue
        a=np.array(v,float)                            # [40,2]
        xs=np.interp(t_dst, t_src, a[:,0]); ys=np.interp(t_dst, t_src, a[:,1])
        out[k]=np.stack([xs,ys],1).tolist()
    json.dump(out, open(path_out,"w"))
    print(f"{name}: {len(out)-1} tokens -> 8pts/0.5s  {path_out}")
resample(f"{CMP}/gtrs.json", f"{CMP}/gtrs_r.json", "GTRS")
resample(f"{CMP}/hydra_mdp.json", f"{CMP}/hydra_mdp_r.json", "HydraMDP")
