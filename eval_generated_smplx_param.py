"""
eval_generated_smplx_param.py
=============================
Generate sign language motion from gloss, optionally render meshes/GIFs,
and evaluate generation quality with model-free and model-based metrics.

Output (generation):
    output_dir/
        GLOSS_NAME/
            GLOSS_NAME_000000_p0.npz
            GLOSS_NAME.gif
            ...

Output (evaluation):
    output_dir/eval_metrics_epXX_YYYYMMDD_HHMMSS.yaml

Usage:
    # Generate + render only
    python eval_generated_smplx_param.py \
        --checkpoint path/to/best_model.pt \
        --render_mesh --gif

    # Generate + evaluate metrics
    python eval_generated_smplx_param.py \
        --checkpoint path/to/best_model.pt \
        --evaluate

    # Full pipeline: generate + render + evaluate
    python eval_generated_smplx_param.py \
        --checkpoint path/to/best_model.pt \
        --render_mesh --gif --evaluate
"""
import os
import argparse
import random
import yaml
from typing import List, Dict, Optional
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from aslAvatarModel import ASLAvatarModel
from aslAvatarModel_v2 import ASLAvatarModelV2
from aslAvatarModel_v4 import ASLAvatarModelV4
from aslAvatarModel_v5 import MotionDiffusionModel

from config import SignBank_SMPLX_Config
from config import WLASL_SMPLX_Config
from config import ASL3DWord_SMPLX_Config

from dataloader.SignBankSMPLXDataset import SignBankSMPLXDataset
from dataloader.WLASLSMPLXDataset import WLASLSMPLXDataset
from dataloader.WLASLSMPLXDatasetV2 import WLASLSMPLXDatasetV2
from dataloader.ASL3DWordDataset import ASL3DWordDataset

from utils.rotation_conversion import postprocess_motion


# =============================================================================
# SMPL-X Parameter Layout (159 dims)
#   root_pose(3) + body_pose(63) + lhand(45) + rhand(45) + jaw(3)
# =============================================================================

PARAM_SLICES = {
    'smplx_root_pose':  (0,   3),
    'smplx_body_pose':  (3,   66),
    'smplx_lhand_pose': (66,  111),
    'smplx_rhand_pose': (111, 156),
    'smplx_jaw_pose':   (156, 159),
}


def split_params(flat: np.ndarray) -> Dict[str, np.ndarray]:
    """Split (159,) vector into named SMPL-X components."""
    return {name: flat[s:e].copy() for name, (s, e) in PARAM_SLICES.items()}


# =============================================================================
# Model Loading
# =============================================================================

def load_model_weight(model, checkpoint_path: str, device: str = 'cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get('model_state_dict', ckpt)
    cur = model.state_dict()
    loaded = 0
    for k in model_state:
        if k in cur and cur[k].shape == model_state[k].shape:
            cur[k] = model_state[k]
            loaded += 1
    model.load_state_dict(cur, strict=False)
    model.to(device).eval()
    print(f"Loaded: {checkpoint_path}  (epoch {ckpt.get('epoch','?')}, {loaded} keys)")
    return model, ckpt


# =============================================================================
# Generation — uses model.generate() with CLIP text conditioning
# =============================================================================

@torch.no_grad()
def generate_from_gloss(model, gloss: str, seq_len: int, device: str, cfg=None):
    """Generate (T, 159) SMPL-X params from gloss via prior sampling."""
    motion = model.generate([gloss], seq_len=seq_len, device=device)

    diff = (motion[0, 1:] - motion[0, :-1]).abs().mean()
    print(f"Generated frame diff: {diff:.8f}")

    motion_raw = motion.squeeze(0).cpu().numpy()  # (T, input_dim)

    if cfg is not None:
        motion_raw = postprocess_motion(motion_raw, cfg)  # (T, 159)

    return motion_raw


@torch.no_grad()
def generate_motion_tensor(model, gloss: str, seq_len: int, device: str, cfg=None):
    """
    Generate motion and return as (T, 53*n_feats) tensor.
    Same as generate_from_gloss but returns tensor, no printing.
    Applies postprocess_motion (converts to 159-dim axis-angle).
    """
    motion = model.generate([gloss], seq_len=seq_len, device=device)
    motion_raw = motion.squeeze(0).cpu().numpy()  # (T, input_dim)
    if cfg is not None:
        motion_raw = postprocess_motion(motion_raw, cfg)  # (T, 159) or (T, 318)
    return torch.tensor(motion_raw, dtype=torch.float32)


@torch.no_grad()
def generate_motion_raw(model, gloss: str, seq_len: int, device: str):
    """
    Generate motion and return RAW model output (no postprocessing).
    Output is in the same representation as the training data,
    so GT and Gen are directly comparable for metrics.

    Returns:
        (T, input_dim) tensor in model's native representation
    """
    motion = model.generate([gloss], seq_len=seq_len, device=device)
    return motion.squeeze(0).cpu()  # (T, input_dim)


# =============================================================================
# Save .npz
# =============================================================================

def save_frame_npz(frame_params: Dict[str, np.ndarray], save_path: str):
    dump = {
        'smplx_root_pose':  frame_params['smplx_root_pose'].reshape(3,).astype(np.float32),
        'smplx_body_pose':  frame_params['smplx_body_pose'].reshape(21, 3).astype(np.float32),
        'smplx_lhand_pose': frame_params['smplx_lhand_pose'].reshape(15, 3).astype(np.float32),
        'smplx_rhand_pose': frame_params['smplx_rhand_pose'].reshape(15, 3).astype(np.float32),
        'smplx_jaw_pose':   frame_params['smplx_jaw_pose'].reshape(3,).astype(np.float32),
        'smplx_shape':      np.zeros(10, dtype=np.float32),
        'smplx_expr':       np.zeros(10, dtype=np.float32),
        'cam_trans':         np.zeros(3, dtype=np.float32),
    }
    np.savez(save_path, **dump)


# =============================================================================
# SMPL-X Mesh Rendering
# =============================================================================

def load_smplx_model(human_model_path):
    """Load SMPL-X model for mesh generation."""
    from human_models.human_models import SMPLX
    smpl_x = SMPLX(human_model_path)
    print(f"Loaded SMPL-X model from {human_model_path}")
    return smpl_x


def params_to_mesh(smpl_x, frame_params):
    """Run SMPL-X forward pass, return (vertices, faces)."""
    def _t(arr):
        return torch.tensor(arr.reshape(1, -1), dtype=torch.float32)

    zeros3 = torch.zeros(1, 3, dtype=torch.float32)
    output = smpl_x.layer['neutral'](
        global_orient=_t(frame_params['smplx_root_pose']),
        body_pose=_t(frame_params['smplx_body_pose']),
        left_hand_pose=_t(frame_params['smplx_lhand_pose']),
        right_hand_pose=_t(frame_params['smplx_rhand_pose']),
        jaw_pose=_t(frame_params['smplx_jaw_pose']),
        leye_pose=zeros3,
        reye_pose=zeros3,
        betas=torch.zeros(1, 10, dtype=torch.float32),
        expression=torch.zeros(1, 10, dtype=torch.float32),
    )
    vertices = output.vertices.cpu().numpy().squeeze(0)
    faces = smpl_x.face.astype(np.int32)
    return vertices, faces


def render_smplx_frame(vertices, faces, img_w=512, img_h=512,
                       debug=False, gloss=""):
    """Render SMPL-X mesh to an image."""
    import trimesh
    import pyrender

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()

    fov_y = np.radians(50.0)
    distance = (extent / 2.0) / np.tan(fov_y / 2.0) * 1.4

    verts_centered = vertices - center

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1, roughnessFactor=0.4, alphaMode="OPAQUE",
        emissiveFactor=(0.15, 0.2, 0.15),
        baseColorFactor=(0.6, 0.9, 0.65, 1.0),
    )

    body_trimesh = trimesh.Trimesh(verts_centered, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=img_w / img_h)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = distance

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=(0.4, 0.4, 0.4))
    scene.add(body_mesh, "mesh")
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose.copy())

    r = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h,
                                   point_size=1.0)
    color_img, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    return color_img[:, :, :3]


# =============================================================================
# Process one gloss (generate + optional render/save)
# =============================================================================

def process_a_gloss(model, gloss, output_dir, seq_len, device,
                    smpl_x=None, img_size=512, dump_param=False,
                    cfg=None, make_gif=True, gif_fps=8, dataset = None):

                
    if cfg.USE_PHONO_ATTRIBUTE:
        gloss_with_attributes = dataset._gloss_with_phono(gloss)
    else:
        gloss_with_attributes = gloss
        
    motion = generate_from_gloss(model, gloss_with_attributes, seq_len, device, cfg)
    T = motion.shape[0]

    gloss_dir = os.path.join(output_dir, gloss)
    os.makedirs(gloss_dir, exist_ok=True)

    if smpl_x is not None:
        render_dir = os.path.join(gloss_dir, 'renders')
        os.makedirs(render_dir, exist_ok=True)

    gif_frames = []
    for t in range(T):
        params = split_params(motion[t])

        if dump_param:
            npz_path = os.path.join(gloss_dir, f"{gloss}_{t:06d}_p0.npz")
            save_frame_npz(params, npz_path)

        if smpl_x is not None:
            try:
                vertices, faces = params_to_mesh(smpl_x, params)
                img = render_smplx_frame(vertices, faces,
                                         img_w=img_size, img_h=img_size,
                                         debug=(t == 0), gloss=gloss)
                if make_gif:
                    gif_frames.append(img)
            except Exception as e:
                import traceback
                print(f"  Render error frame {t}: {e}")
                traceback.print_exc()

    if make_gif and gif_frames:
        gif_path = os.path.join(gloss_dir, f"{gloss}.gif")
        save_gif(gif_frames, gif_path, fps=gif_fps)

    return T


# =============================================================================
# Helpers
# =============================================================================

def get_glosses_from_dataset(root_dir: str, num_glosses: Optional[int] = None) -> List[str]:
    if not os.path.isdir(root_dir):
        print(f"WARNING: not found: {root_dir}")
        return []
    glosses = sorted(d for d in os.listdir(root_dir)
                     if os.path.isdir(os.path.join(root_dir, d)))
    if num_glosses and num_glosses < len(glosses):
        glosses = random.sample(glosses, num_glosses)
    return glosses


def save_gif(frames, gif_path, fps=10):
    import imageio
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {gif_path} ({len(frames)} frames, {fps} fps)")


def make_collate_fn(gloss_to_idx):
    """
    Create a collate function that converts ASL3DWordDataset's
    (seq, gloss_str, actual_len) tuples into {"x": ..., "y": ...} batches.
    """
    def collate_fn(batch_list):
        # batch_list: list of (seq_tensor, gloss_str, actual_len)
        seqs = [item[0] for item in batch_list]       # list of (T, input_dim)
        glosses = [item[1] for item in batch_list]     # list of str
        gloss_with_attributes = [item[2] for item in batch_list]     # list of str
        
        lengths = len(seqs)
        
        # motion, gloss, gloss_with_attributes = batch

        # Stack sequences (already padded to target_seq_len by dataset)
        x = torch.stack(seqs, dim=0)  # (B, T, input_dim)

        # Convert gloss strings → label indices
        y = torch.tensor([gloss_to_idx[g] for g in glosses], dtype=torch.long)

        return {"x": x, "y": y, "lengths": torch.tensor(lengths, dtype=torch.long),
                "glosses": glosses}

    return collate_fn


# =============================================================================
# Evaluation: collect GT and Gen motions, run metrics
# =============================================================================

@torch.no_grad()
def collect_gt_motions(dataset, batch_size=32):
    """
    Collect all GT motions from a dataset.

    Returns:
        motions: (N, T, input_dim)   where input_dim = 53*n_feats
        labels:  (N,)
    """
    from torch.utils.data import DataLoader

    collate_fn = make_collate_fn(dataset.gloss_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_fn)
    all_x, all_y = [], []
    for batch in tqdm(loader, desc="Collecting GT"):
        all_x.append(batch["x"])
        all_y.append(batch["y"])
    return torch.cat(all_x, 0), torch.cat(all_y, 0)


@torch.no_grad()
def collect_gen_motions(model, dataset, cfg, device, batch_size=32):
    """
    For each GT sample, generate motion with the same class label.

    Returns:
        motions: (N, T, input_dim)
        labels:  (N,)
    """
    from torch.utils.data import DataLoader

    collate_fn = make_collate_fn(dataset.gloss_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_fn)
    all_x, all_y = [], []

    for batch in tqdm(loader, desc="Generating"):
        y = batch["y"]
        glosses = batch["glosses"]
        B = y.shape[0]
        gen_list = []
        for i in range(B):
            gloss = glosses[i]
            # Use raw output (no postprocessing) to match GT representation
            motion_t = generate_motion_raw(
                model, gloss, cfg.TARGET_SEQ_LEN, device)
            gen_list.append(motion_t)

        all_x.append(torch.stack(gen_list, 0))
        all_y.append(y)

    return torch.cat(all_x, 0), torch.cat(all_y, 0)


def run_evaluation(model, cfg, device, train_dataset, test_dataset,
                   batch_size=32, seed=42):
    """
    Full evaluation pipeline:
      1. Collect GT train + test
      2. Generate motions for test set
      3. Run model-free metrics
      4. Run model-based metrics (train MLP on GT train, evaluate on test)

    Returns:
        dict with all results
    """
    from model_free_metrics import ModelFreeEvaluator
    from model_based_metrics import ModelBasedEvaluator

    n_feats = cfg.N_FEATS
    num_classes = cfg.NUM_CLASSES

    # ── Align label indices: remap test labels to train's gloss_to_idx ──
    # Both datasets build gloss_to_idx independently, so indices may differ.
    # We use train's mapping as the canonical one.
    train_g2i = train_dataset.gloss_to_idx
    test_g2i  = test_dataset.gloss_to_idx

    # Build test→train label remapping
    test_to_train_label = {}
    missing_glosses = set()
    for gloss, test_idx in test_g2i.items():
        if gloss in train_g2i:
            test_to_train_label[test_idx] = train_g2i[gloss]
        else:
            missing_glosses.add(gloss)
    if missing_glosses:
        print(f"  WARNING: {len(missing_glosses)} test glosses not in train: "
              f"{missing_glosses}")

    # --- Collect data ---
    print("\n>>> Collecting GT train motions ...")
    gt_train_x, gt_train_y = collect_gt_motions(train_dataset, batch_size)
    print(f"    GT train: {gt_train_x.shape}, labels range: "
          f"[{gt_train_y.min()}, {gt_train_y.max()}]")

    print("\n>>> Collecting GT test motions ...")
    gt_test_x, gt_test_y_raw = collect_gt_motions(test_dataset, batch_size)
    print(f"    GT test: {gt_test_x.shape}")

    # Remap test labels to train's index space
    gt_test_y = gt_test_y_raw.clone()
    valid_mask = torch.ones(len(gt_test_y), dtype=torch.bool)
    for i in range(len(gt_test_y)):
        old_idx = gt_test_y_raw[i].item()
        if old_idx in test_to_train_label:
            gt_test_y[i] = test_to_train_label[old_idx]
        else:
            valid_mask[i] = False

    if not valid_mask.all():
        n_removed = (~valid_mask).sum().item()
        print(f"    Removing {n_removed} test samples with unknown glosses")
        gt_test_x = gt_test_x[valid_mask]
        gt_test_y = gt_test_y[valid_mask]
    print("\n>>> Generating motions for test set ...")
    gen_test_x, gen_test_y_raw = collect_gen_motions(
        model, test_dataset, cfg, device, batch_size)
    print(f"    Gen test: {gen_test_x.shape}")

    # Remap gen labels too
    gen_test_y = gen_test_y_raw.clone()
    valid_mask_gen = torch.ones(len(gen_test_y), dtype=torch.bool)
    for i in range(len(gen_test_y)):
        old_idx = gen_test_y_raw[i].item()
        if old_idx in test_to_train_label:
            gen_test_y[i] = test_to_train_label[old_idx]
        else:
            valid_mask_gen[i] = False

    if not valid_mask_gen.all():
        gen_test_x = gen_test_x[valid_mask_gen]
        gen_test_y = gen_test_y[valid_mask_gen]

    # --- Model-free metrics ---
    print("\n" + "="*65)
    print("  Running Model-Free Evaluation")
    print("="*65)
    mf_eval = ModelFreeEvaluator(
        n_feats=n_feats, num_classes=num_classes, seed=seed)
    mf_results = mf_eval.evaluate(gt_test_x, gt_test_y,
                                   gen_test_x, gen_test_y)
    mf_eval.print_results(mf_results, title="test")

    # --- Model-based metrics ---
    print("\n" + "="*65)
    print("  Running Model-Based Evaluation")
    print("="*65)
    mb_eval = ModelBasedEvaluator(
        n_feats=n_feats, num_classes=num_classes,
        device=device, seed=seed)

    # Train classifier on GT train
    mb_eval.train(gt_train_x, gt_train_y)

    # Evaluate on test
    mb_results = mb_eval.evaluate(gt_test_x, gt_test_y,
                                   gen_test_x, gen_test_y)
    mb_eval.print_results(mb_results, title="test")

    return {
        "model_free": mf_results,
        "model_based": mb_results,
    }


# =============================================================================
# Dataset loading helper
# =============================================================================

def load_dataset(dataset_name, cfg, mode='train', logger=None):
    """Load dataset by name and mode."""
    if dataset_name == "SignBank_SMPLX":
        return SignBankSMPLXDataset(mode=mode, cfg=cfg)
    elif dataset_name == "WLASL_SMPLX":
        ver = getattr(cfg, 'DATASET_VERSION', 'v1').lower()
        if ver == 'v2':
            return WLASLSMPLXDatasetV2(mode=mode, cfg=cfg, logger=logger)
        else:
            return WLASLSMPLXDataset(mode=mode, cfg=cfg, logger=logger)
    elif dataset_name == "ASL3DWord":
        return ASL3DWordDataset(mode=mode, cfg=cfg, logger=logger)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# =============================================================================
# Model creation helper
# =============================================================================

def create_model(cfg):
    """Create generation model based on config."""
    ver = cfg.MODEL_VERSION.lower()
    if ver == 'v1':
        return ASLAvatarModel(cfg)
    elif ver == 'v2':
        return ASLAvatarModelV2(cfg)
    elif ver == 'v4':
        return ASLAvatarModelV4(cfg)
    elif ver == 'v5':
        return MotionDiffusionModel(cfg)
    else:
        raise ValueError(f"Unknown model version: {cfg.MODEL_VERSION}")


# =============================================================================
# Main
# =============================================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---- Config ----
    config_map = {
        "SignBank_SMPLX": SignBank_SMPLX_Config,
        "WLASL_SMPLX":   WLASL_SMPLX_Config,
        "ASL3DWord":      ASL3DWord_SMPLX_Config,
    }
    cfg = config_map[args.dataset_name]()
    cfg.USE_UPPER_BODY    = args.use_upper_body
    cfg.USE_ROT6D         = args.use_rot6d
    cfg.USE_MINI_DATASET  = args.use_mini_dataset
    cfg.ROOT_NORMALIZE    = not args.no_root_normalize
    cfg.N_FEATS           = 6 if cfg.USE_ROT6D else 3
    cfg.USE_PHONO_ATTRIBUTE = args.use_phono_attribute
    cfg.TEXT_ENCODER_TYPE = args.text_encoder_type

    # ---- Dataset ----
    train_dataset = load_dataset(args.dataset_name, cfg, mode='train')
    test_dataset = None
    if args.evaluate or args.dataset_name == "ASL3DWord":
        try:
            test_dataset = load_dataset(args.dataset_name, cfg, mode='test')
        except Exception as e:
            print(f"WARNING: Could not load test dataset: {e}")

    cfg.INPUT_DIM       = train_dataset.input_dim
    cfg.GLOSS_NAME_LIST = train_dataset.gloss_name_list
    cfg.NUM_CLASSES     = len(cfg.GLOSS_NAME_LIST)

    print(f"Dataset: {args.dataset_name}, {cfg.NUM_CLASSES} classes, "
          f"n_feats={cfg.N_FEATS}")
    print(f"  train: {len(train_dataset)} samples")
    if test_dataset:
        print(f"  test:  {len(test_dataset)} samples")

    # ---- Model ----
    model = create_model(cfg)
    model, ckpt = load_model_weight(model, args.checkpoint, device)
    epoch_str = ckpt.get('epoch', 'unknown')

    seq_len = cfg.TARGET_SEQ_LEN

    # ---- Output dir ----
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        logging_dir = checkpoint_dir.replace(
            '/scratch/rhong5/weights/temp_training_weights/aslAvatar',
            '/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/zlog')
        os.makedirs(logging_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_log_dir = os.path.join(logging_dir, f"test_{timestamp}")
        output_dir = os.path.join(test_log_dir, 'gen_images')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_log_dir = os.path.join(args.output_dir, f"test_{timestamp}")
        output_dir = os.path.join(test_log_dir, 'gen_images')

    os.makedirs(output_dir, exist_ok=True)

    # ==================================================================
    # Phase 1: Generation + Rendering (original functionality)
    # ==================================================================
    if not args.eval_only:
        glosses = args.glosses if args.glosses else cfg.GLOSS_NAME_LIST

        smpl_x = None
        if args.render_mesh:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            smpl_x = load_smplx_model(cfg.HUMAN_MODELS_PATH)

        print(f"\nGenerating {len(glosses)} glosses, {seq_len} frames each")
        if smpl_x:
            print(f"Mesh rendering enabled ({args.img_size}x{args.img_size})")

        total = 0
        for gloss in tqdm(glosses, desc="Generating"):
            total += process_a_gloss(
                model, gloss, output_dir, seq_len, device,
                smpl_x=smpl_x, img_size=args.img_size,
                dump_param=args.dump_param,
                cfg=cfg,
                make_gif=args.gif,
                gif_fps=args.gif_fps,
                dataset=train_dataset,
            )
        print(f"\nDone! {len(glosses)} glosses, {total} frames -> {output_dir}")

    # ==================================================================
    # Phase 2: Evaluation
    # ==================================================================
    if args.evaluate:
        if test_dataset is None:
            print("\nERROR: Cannot evaluate without a test dataset.")
            print("  Use --dataset_name ASL3DWord or a dataset with test split.")
            return

        print("\n" + "#"*65)
        print("  EVALUATION")
        print("#"*65)

        all_results = run_evaluation(
            model, cfg, device,
            train_dataset, test_dataset,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # ---- Save results ----
        save_dir = test_log_dir
        os.makedirs(save_dir, exist_ok=True)

        # Serialize variance_ratio nested dicts
        def _serialize(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = _serialize(v)
                elif isinstance(v, (np.floating, np.integer)):
                    out[k] = float(v)
                elif isinstance(v, float):
                    out[k] = round(v, 6)
                else:
                    out[k] = v
            return out

        save_data = {
            "config": {
                "checkpoint": args.checkpoint,
                "epoch": str(epoch_str),
                "dataset": args.dataset_name,
                "num_classes": cfg.NUM_CLASSES,
                "n_feats": cfg.N_FEATS,
                "use_upper_body": args.use_upper_body,
                "use_rot6d": args.use_rot6d,
                "root_normalize": cfg.ROOT_NORMALIZE,
                "seed": args.seed,
            },
            "model_free": _serialize(all_results["model_free"]),
            "model_based": _serialize(all_results["model_based"]),
        }

        fname = f"eval_metrics_ep{epoch_str}_{timestamp}.yaml"
        save_path = os.path.join(save_dir, fname)
        with open(save_path, 'w') as f:
            yaml.dump(save_data, f, default_flow_style=False, sort_keys=False)
        print(f"\n>>> Metrics saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SMPL-X sign motion + evaluate quality")

    # Model / Data
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="WLASL_SMPLX",
                        choices=["SignBank_SMPLX", "WLASL_SMPLX", "ASL3DWord"])
    parser.add_argument("--glosses", type=str, nargs='+', default=None,
                        help="Specific glosses to generate (default: all)")

    # Model config
    parser.add_argument("--use_upper_body", action="store_true",
                        help="Model trained on upper body only")
    parser.add_argument("--use_rot6d", action="store_true",
                        help="Model uses 6D rotation representation")
    parser.add_argument("--no_root_normalize", action="store_true", default=False,
                        help="Disable root pose normalization")
    parser.add_argument("--use_mini_dataset", action="store_true",
                        help="Use mini dataset for debugging")

    # Generation
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--render_mesh", action="store_true",
                        help="Render SMPL-X meshes")
    parser.add_argument("--dump_param", action="store_true",
                        help="Save per-frame .npz parameters")
    parser.add_argument("--gif", action="store_true",
                        help="Generate animated GIF per gloss")
    parser.add_argument("--gif_fps", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)

    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Run model-free + model-based evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip generation, only run evaluation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation data loading")
    parser.add_argument("--use_phono_attribute", action="store_true", default=False)
    parser.add_argument("--text_encoder_type", type=str, default='clip', choices=["clip", "t5"])

    # General
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # eval_only implies evaluate
    if args.eval_only:
        args.evaluate = True

    main(args)