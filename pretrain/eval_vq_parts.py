"""Eval-only (2026-09-18): per-part reconstruction error of one or more PartVQVAE checkpoints on the SAME val windows, with the v2 logger
(mpjpe overall / arm (14 body+arm joints) / finger (30 joints), per-part rec MSE, codebook perplexity). Needed because v1 training did not log
arm and finger mpjpe separately. Loss weights are fixed to 1 here, so numbers are comparable across checkpoints.
  python pretrain/eval_vq_parts.py --val_list .../pool_val.txt --ckpts v1=<pt> v2=<pt> --out <json>
"""
import argparse, json, os, sys
import torch
from torch.utils.data import DataLoader
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.motion_window_dataset import MotionWindowDataset, collate
from pretrain.part_vqvae import PartVQVAE
from pretrain.train_part_vqvae_v2 import losses, INDEX
from utils.smplx_fk_diff_fast import SMPLXForwardKinematicsFast


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--index', default=INDEX); ap.add_argument('--val_list', required=True); ap.add_argument('--ckpts', nargs='+', required=True, help='name=path')
    ap.add_argument('--out', required=True); ap.add_argument('--batch_size', type=int, default=256); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(); dev = 'cuda'
    fk = SMPLXForwardKinematicsFast().to(dev).eval(); res = {}
    for item in a.ckpts:
        name, path = item.split('=', 1); ck = torch.load(path, map_location='cpu'); cfg = ck['config']
        w = argparse.Namespace(w_hand=1.0, w_face=1.0, fk_w_finger=1.0, lambda_vel=cfg['lambda_vel'], beta=cfg['beta'], lambda_fk=cfg['lambda_fk'])
        model = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); model.load_state_dict(ck['model']); model.eval()
        dv = DataLoader(MotionWindowDataset(a.index, a.val_list, cfg['window'], False, None), a.batch_size, shuffle=False, num_workers=a.workers, collate_fn=collate)
        vs, n = {}, 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for vb in dv:
                _, s = losses(model, vb, fk, w, dev); n += 1
                for k, v in s.items(): vs[k] = vs.get(k, 0) + v
        res[name] = {'ckpt': path, 'step': ck.get('step'), **{k: v / n for k, v in vs.items()}}
        print(name, json.dumps({k: round(v, 4) for k, v in res[name].items() if isinstance(v, float)}), flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1); print('wrote', a.out)


if __name__ == '__main__': main()
