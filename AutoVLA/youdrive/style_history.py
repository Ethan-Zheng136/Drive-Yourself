"""History<->Future style coupling.

Extracts style features from the PAST window (ego history frames) and correlates
them with the FUTURE window (styletest). High correlation => recent driving
predicts near-future style => history is a usable style prior / consistency signal.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from style_metrics import _features, human_features
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

DATA="/root/workspace/closed_loop/data/navsim"
ST=json.load(open("/root/workspace/tools/scorer/data/styledrive/styletest.json"))
DT=0.5
FEATS=["v_avg","peak_decel","jerk_rms","lat_vmax"]
CAP=int(os.environ.get("CAP","1500"))

def hist_features(scene):
    n=scene.scene_metadata.num_history_frames
    vx=[]; vy=[]
    for i in range(n):
        ev=scene.frames[i].ego_status.ego_velocity
        vx.append(float(ev[0])); vy.append(float(ev[1]))
    if len(vx)<2: return None
    return _features(np.array(vx),np.array(vy),DT)

def run(split):
    sf=SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,
                   has_route=True,max_scenes=None,log_names=None,tokens=None)
    loader=SceneLoader(data_path=Path(f"{DATA}/navsim_logs/{split}"),
                       sensor_blobs_path=Path(f"{DATA}/sensor_blobs/{split}"),
                       scene_filter=sf,sensor_config=SensorConfig.build_no_sensors())
    common=[t for t in loader.tokens if t in ST]
    print(f"[{split}] loader tokens={len(loader.tokens)}  ∩ styletest={len(common)}")
    if not common: return None
    common=common[:CAP]
    H={k:[] for k in FEATS}; F={k:[] for k in FEATS}; rows=[]
    for tok in common:
        try:
            sc=loader.get_scene_from_token(tok)
            hf=hist_features(sc)
            ff=human_features(ST[tok])
            if hf is None or ff is None: continue
            ok=all(not math.isnan(hf[k]) and not math.isnan(ff[k]) for k in FEATS)
            if not ok: continue
            for k in FEATS: H[k].append(hf[k]); F[k].append(ff[k])
            rows.append((tok,hf,ff))
        except Exception as e:
            print("skip",tok,repr(e)[:80])
    n=len(rows); print(f"[{split}] usable pairs={n}")
    if n<10: return None
    res={"split":split,"n":n,"pearson":{}}
    for k in FEATS:
        a=np.array(H[k]); b=np.array(F[k])
        r=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else float("nan")
        res["pearson"][k]=r
        print(f"  corr(hist,future) {k:<12} r={r:+.3f}")
    res["_arrays"]={k:[H[k],F[k]] for k in FEATS}
    return res

if __name__=="__main__":
    out=None
    for sp in ["test","trainval"]:
        try:
            r=run(sp)
            if r: out=r; break
        except Exception as e:
            print(f"[{sp}] FAILED {e!r}"[:160])
    if out:
        json.dump({k:v for k,v in out.items() if k!="_arrays"}|{"arrays":out["_arrays"]},
                  open("youdrive/style_history_out.json","w"))
        print("\nwritten youdrive/style_history_out.json")
    else:
        print("NO usable data on any split"); sys.exit(1)
