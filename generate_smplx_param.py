"""
SignBank SMPL-X Inference Script (params only)

Generate sign language motion from gloss, save SMPL-X parameters (.npz) per frame.

Output:
    output_dir/
        GLOSS_NAME/
            GLOSS_NAME_000000_p0.npz
            GLOSS_NAME_000001_p0.npz
            ...

Usage:
    python generate_smplx_param.py \
        --checkpoint path/to/best_model.pt \
        --glosses AMAZING HELLO THANK-YOU

    python generate_smplx_param.py \
        --checkpoint path/to/best_model.pt \
        --from_dataset --num_glosses 20
"""
import logging

import os
import argparse
import random
from typing import List, Dict, Optional
from accelerate.logging import get_logger

import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from aslAvatarModel import ASLAvatarModel
from aslAvatarModel_v2 import ASLAvatarModelV2
from aslAvatarModel_v3 import ASLAvatarModelV3
from aslAvatarModel_v4 import ASLAvatarModelV4
from aslAvatarModel_v5 import MotionDiffusionModel

from config import SignBank_SMPLX_Config
from config import WLASL_SMPLX_Config
from config import ASL3DWord_SMPLX_Config

# from dataloader.WLASLSMPLXDataset import WLASLSMPLXDataset
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
    return model


# =============================================================================
# Generation — uses model.generate() (sample from prior N(0,I))
# =============================================================================

@torch.no_grad()
def generate_from_gloss(model, gloss: str, seq_len: int, device: str, cfg=None):
    """Generate (T, 159) SMPL-X params from gloss via prior sampling."""
    if cfg.USE_LABEL_INDEX_COND:
        gloss_index = cfg.GLOSS_NAME_LIST.index(gloss)
        label_indices = torch.tensor([gloss_index], dtype=torch.long).to(device)
        motion = model.generate(label_indices, seq_len=seq_len, device=device)
    else:
        motion = model.generate([gloss], seq_len=seq_len, device=device)

    diff = (motion[0, 1:] - motion[0, :-1]).abs().mean()
    print(f"Generated frame diff: {diff:.8f}")

    motion_raw = motion.squeeze(0).cpu().numpy()  # (T, input_dim)

    if cfg is not None:
        motion_raw = postprocess_motion(motion_raw, cfg)  # (T, 159)

    return motion_raw



# =============================================================================
# Save .npz (matching your extraction pipeline format)
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
    """
    Render SMPL-X mesh to an image. Self-contained, no external camera functions.
    
    Args:
        vertices: (N, 3) mesh vertices
        faces: (F, 3) face indices
        img_w, img_h: output image size
        debug: print camera/scene info
        gloss: for debug prints
    
    Returns:
        img: (H, W, 3) uint8 RGB image
    """
    import trimesh
    import pyrender

    # --- 1. Compute camera to frame the mesh ---
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()

    # Camera distance so mesh fills ~60% of frame
    fov_y = np.radians(50.0)
    distance = (extent / 2.0) / np.tan(fov_y / 2.0) * 1.4

    if debug:
        print(f"  [{gloss}] center={center}, extent={extent:.4f}, distance={distance:.4f}")

    # --- 2. Center the mesh at origin ---
    verts_centered = vertices - center

    if debug:
        vmin2 = verts_centered.min(axis=0)
        vmax2 = verts_centered.max(axis=0)
        print(f"  [{gloss}] centered verts: min={vmin2}, max={vmax2}")

    # --- 3. Build scene ---
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.4,
        alphaMode="OPAQUE",
        emissiveFactor=(0.15, 0.2, 0.15),
        baseColorFactor=(0.6, 0.9, 0.65, 1.0),
    )

    body_trimesh = trimesh.Trimesh(verts_centered, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # Perspective camera
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=img_w / img_h)

    # Camera pose: placed at (0, 0, distance) looking at origin
    # pyrender camera looks along -Z in its local frame
    cam_pose = np.eye(4)
    cam_pose[2, 3] = distance  # move camera back along +Z

    if debug:
        print(f"  [{gloss}] cam_pose:\n{cam_pose}")

    # Lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_pose = cam_pose.copy()  # same position as camera

    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],  # white background
        ambient_light=(0.4, 0.4, 0.4),
    )
    scene.add(body_mesh, "mesh")
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=light_pose)

    # --- 4. Render ---
    r = pyrender.OffscreenRenderer(
        viewport_width=img_w,
        viewport_height=img_h,
        point_size=1.0,
    )
    color_img, depth_img = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    if debug:
        depth_valid = depth_img[depth_img > 0]
        print(f"  [{gloss}] render RGBA: shape={color_img.shape}, "
              f"dtype={color_img.dtype}, min={color_img.min()}, max={color_img.max()}")
        print(f"  [{gloss}] depth: valid_pixels={len(depth_valid)}, "
              f"range=[{depth_valid.min():.3f}, {depth_valid.max():.3f}]" 
              if len(depth_valid) > 0 else f"  [{gloss}] depth: NO valid pixels (mesh not visible!)")
        alpha_nonzero = (color_img[:, :, 3] > 0).sum()
        print(f"  [{gloss}] alpha > 0 pixels: {alpha_nonzero} / {img_w * img_h}")

    return color_img[:, :, :3]  # (H, W, 3) uint8



# =============================================================================
# Process one gloss
# =============================================================================
def process_a_gloss(model, gloss, output_dir, seq_len, device,
                  smpl_x=None, img_size=512, dump_param = False, 
                        cfg=None, make_gif=True, gif_fps=8, dataset = None):

    if cfg.USE_PHONO_ATTRIBUTE:
        gloss_str = dataset._gloss_with_phono(gloss)
    else:
        gloss_str = gloss
        
    motion = generate_from_gloss(model, gloss_str, seq_len, device, cfg)  # (T, 159)
    print('motion.shape', motion.shape, 'cfg.N_FEATS', cfg.N_FEATS)
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
            # Save .npz
            npz_path = os.path.join(gloss_dir, f"{gloss}_{t:06d}_p0.npz")
            save_frame_npz(params, npz_path)

        # Render mesh
        if smpl_x is not None:
            try:
                import cv2
                vertices, faces = params_to_mesh(smpl_x, params)

                # Debug: print mesh stats for first frame
                # if t == 0:
                #     print(f"  [{gloss}] vertices: shape={vertices.shape}, "
                #           f"min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")
                #     print(f"  [{gloss}] faces: shape={faces.shape}, "
                #           f"min={faces.min()}, max={faces.max()}")
                #     has_nan = np.any(np.isnan(vertices))
                #     has_inf = np.any(np.isinf(vertices))
                #     print(f"  [{gloss}] NaN={has_nan}, Inf={has_inf}")

                img = render_smplx_frame(vertices, faces, 
                                         img_w=img_size, img_h=img_size,
                                         debug=(t == 0), gloss=gloss)
                img_path = os.path.join(render_dir, f"{gloss}_{t:06d}.png")
                # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
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
# Gloss discovery from dataset dir
# =============================================================================

def get_glosses_from_dataset(root_dir: str, num_glosses: Optional[int] = None) -> List[str]:
    if not os.path.isdir(root_dir):
        print(f"WARNING: not found: {root_dir}")
        return []
    glosses = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    if num_glosses and num_glosses < len(glosses):
        glosses = random.sample(glosses, num_glosses)
    return glosses

def save_gif(frames, gif_path, fps=10):
    """
    Save a list of RGB numpy arrays as an animated GIF.
    
    Args:
        frames: list of (H, W, 3) uint8 numpy arrays
        gif_path: output .gif path
        fps: frames per second
    """
    import imageio
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {gif_path} ({len(frames)} frames, {fps} fps)")


# =============================================================================
# Main
# =============================================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = 'cuda'
    if args.dataset_name == "SignBank_SMPLX":
        Config = SignBank_SMPLX_Config
    elif args.dataset_name == "WLASL_SMPLX":
        Config = WLASL_SMPLX_Config
    elif args.dataset_name == 'ASL3DWord':
        Config = ASL3DWord_SMPLX_Config
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    cfg = Config()
    cfg.USE_UPPER_BODY = args.use_upper_body
    cfg.USE_ROT6D = args.use_rot6d
    cfg.USE_MINI_DATASET = args.use_mini_dataset
    cfg.USE_LABEL_INDEX_COND= args.use_label_index_cond
    cfg.ROOT_NORMALIZE = not args.no_root_normalize
    cfg.N_FEATS = 6 if cfg.USE_ROT6D else 3
    cfg.USE_PHONO_ATTRIBUTE = args.use_phono_attribute
    
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)        
        logging_dir = checkpoint_dir.replace('/scratch/rhong5/weights/temp_training_weights', '/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/zlog')
        os.makedirs(logging_dir, exist_ok=True)
        
        # Output dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_log_dir = os.path.join(logging_dir, f"test_{timestamp}")
        output_dir = os.path.join(logging_dir, f"test_{timestamp}", 'gen_images')        
    else:
        logging_dir = args.output_dir
        test_log_dir = os.path.join(logging_dir, f"test_{timestamp}")
        output_dir = os.path.join(logging_dir, f"test_{timestamp}", 'gen_images')        

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(test_log_dir, 'test.log')

    # handlers = [
    #     logging.FileHandler(log_file),
    #     logging.StreamHandler()
    #     ]

    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     level=logging.INFO,
    #     handlers=handlers
    # )
    
    # logger = get_logger(__name__)
    logger = None
        
    if cfg.DATASET_NAME == "SignBank_SMPLX":
        train_dataset = SignBankSMPLXDataset(mode='train', cfg=cfg)
        test_dataset = None  # TODO: implement test dataset for SignBank_SMPLX
    elif cfg.DATASET_NAME == "WLASL_SMPLX":
        if cfg.DATASET_VERSION.lower() == 'v1':
            train_dataset = WLASLSMPLXDataset(mode='train', cfg=cfg, logger=logger)
            # test_dataset = WLASLSMPLXDataset(mode='test', cfg=cfg)
        elif cfg.DATASET_VERSION.lower() == 'v2':
            train_dataset = WLASLSMPLXDatasetV2(mode='train', cfg=cfg, logger=logger)
            # test_dataset = WLASLSMPLXDatasetV2(mode='test', cfg=cfg)
    elif cfg.DATASET_NAME == "ASL3DWord":
        if cfg.DATASET_VERSION.lower() == 'v1':
            train_dataset = ASL3DWordDataset(mode='train', cfg=cfg, logger=logger)
            test_dataset = ASL3DWordDataset(mode='test', cfg=cfg, logger=logger)

    else:
        raise ValueError(f"Unknown dataset: {cfg.DATASET_NAME}")
    
    cfg.INPUT_DIM = train_dataset.input_dim

    cfg.INPUT_DIM = train_dataset.input_dim
    cfg.GLOSS_NAME_LIST = train_dataset.gloss_name_list
    cfg.NUM_CLASSES = len(cfg.GLOSS_NAME_LIST)
    
    # Model

    if not cfg.USE_LABEL_INDEX_COND:
        if cfg.MODEL_VERSION.lower() == 'v1':
            model = ASLAvatarModel(cfg)
        elif cfg.MODEL_VERSION.lower() == 'v2':
            model = ASLAvatarModelV2(cfg)
        elif cfg.MODEL_VERSION.lower() == 'v4':
            model = ASLAvatarModelV4(cfg)
        elif cfg.MODEL_VERSION.lower() == 'v5':
            model = MotionDiffusionModel(cfg)
        else:
            raise ValueError('incorrect model version!')
    else:
        model = ASLAvatarModelV3(cfg)

        

    model = load_model_weight(model, args.checkpoint, device)

    seq_len = cfg.TARGET_SEQ_LEN

    # Determine glosses
    if args.glosses:
        glosses = args.glosses
    else:
        glosses = cfg.GLOSS_NAME_LIST
   

    # Load SMPL-X model if rendering
    smpl_x = None
    if args.render_mesh:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        smpl_x = load_smplx_model(cfg.HUMAN_MODELS_PATH)

    print(f"Generating {len(glosses)} glosses, {seq_len} frames each")
    if smpl_x:
        print(f"Mesh rendering enabled ({args.img_size}x{args.img_size})")



    total = 0
    for gloss in tqdm(glosses, desc="Generating"):
        total += process_a_gloss(
            model, gloss, output_dir, seq_len, device,
            smpl_x=smpl_x, img_size=args.img_size,
            dump_param=args.dump_param,
            cfg = cfg,
            # make_gif=args.gif,
            gif_fps=args.gif_fps,
            dataset=train_dataset,
        )



    print(f"\nDone! {len(glosses)} glosses, {total} frames -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-X Generation (params only)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="WLASL_SMPLX", choices=["SignBank_SMPLX", "WLASL_SMPLX", "ASL3DWord"], help="Dataset to use")
    parser.add_argument("--glosses", type=str, nargs='+', default=None, help="e.g. --glosses amazing hello")
    parser.add_argument("--from_dataset", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_glosses", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--render_mesh", action="store_true", help="Enable render mesh")
    parser.add_argument("--dump_param", action="store_true", help="Enable dump parameters")
    parser.add_argument("--use_upper_body", action="store_true", help="only use upper body")
    parser.add_argument("--use_rot6d", action="store_true", help="use 6d rotation")
    parser.add_argument("--img_size", type=int, default=512,help="Output image size (square)")
    parser.add_argument("--use_mini_dataset", action="store_true", help="use mini dataset")
    parser.add_argument("--use_label_index_cond", action="store_true", help = "use label index condition")
    parser.add_argument("--no_root_normalize", action="store_true", default=False, help="Disable root pose normalization (subtract first frame root)")
    parser.add_argument("--gif", action="store_true", help="Generate animated GIF per gloss (requires --render_mesh)")
    parser.add_argument("--gif_fps", type=int, default=8, help="GIF frame rate (default: 10)")
    parser.add_argument("--use_phono_attribute", action="store_true", default=False)


    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
