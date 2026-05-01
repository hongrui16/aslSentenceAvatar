


class BaseConfig:
    def __init__(self):
        
        # ==================== CLIP (Condition Encoder) ====================
        self.CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
        self.CLIP_DIM = 512
        
        # ==================== Model Architecture ====================
        self.LATENT_DIM = 256
        self.MODEL_DIM = 512
        self.N_HEADS = 8
        self.N_LAYERS = 4
        self.DROPOUT = 0.1
        
        # ==================== Training ====================
        self.TRAIN_BSZ = 200
        self.EVAL_BSZ = 200
        self.GRAD_ACCUM = 1
        self.MAX_EPOCHS = 100
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 0.01

        
        # Loss weights
        self.KL_WEIGHT = 1e-3  # Start small to avoid posterior collapse
        
        # Curriculum Learning
        self.USE_CURRICULUM = True
        self.MAX_MASK_RATIO = 0.6
        self.MASK_STEP_EPOCHS = 50
        self.MASK_INCREMENT = 0.1
        
        # ==================== Hardware ====================
        self.MIXED_PRECISION = "fp16"
        self.NUM_WORKERS = 4
        # Preload all per-frame pkls into RAM in dataset.__init__.
        # Saves ~30-100× per-batch IO at the cost of ~2-3GB resident memory.
        self.PRELOAD_TO_MEMORY = False
        # Workers used during the one-time preload (reuses CPU budget; not
        # active during training, so DataLoader's NUM_WORKERS isn't competing).
        self.PRELOAD_WORKERS = 5
        
        # ==================== Logging & Checkpoints ====================
        self.PROJECT_NAME = "ASLSenAvatar"
        self.LOG_INTERVAL = 50
        self.EVAL_INTERVAL = 5  # Evaluate every N epochs

        
        # Paths (adjust to your environment)
        self.LOG_DIR = "/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog"
        self.CKPT_DIR = "/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar"
        
        
        self.RESUME = None  # Path to checkpoint for resuming training
        self.FINETUNE = False  # Whether to finetune (load weights but reset training state)

        self.HUMAN_MODELS_PATH = "./human_models/human_model_files"
        
        self.MODEL_VERSION = 'v1'        
        self.DATASET_VERSION = 'v1'
        
        self.USE_ROT6D = False
        self.USE_UPPER_BODY = False

        self.USE_MINI_DATASET = False
        
        self.USE_LABEL_INDEX_COND = False
        
        self.GLOSS_NAME_LIST = []

        self.NUM_CLASSES = None
        self.EMBED_DIM = 256
        
        self.VEL_WEIGHT  = 1.0
        self.ROOT_NORMALIZE = True
        
        self.N_JOINTS = 53
        self.N_FEATS = 3
        self.USE_PHONO_ATTRIBUTE = False
        
        self.TEXT_ENCODER_TYPE = 'clip'
        
        self.GNN_JOINT_DIM = 128
        self.GNN_N_LAYERS = 4

        # ==================== Voting Module ====================
        self.GLOSS_SOURCE = 'rule_based'  # 'rule_based' or 'llm_draft'
        self.GLOSS_ENCODING = 'per_word'  # 'per_word' (default) or 'whole_str'
        self.VOTING_N_LAYERS = 2
        self.VOTING_N_HEADS = 4
        self.VOTING_FF_MULT = 2
        self.VOTING_MAX_WORDS = 64
        self.SPARSE_WEIGHT = 0.01
        self.ENTROPY_WEIGHT = 0.01

        # ==================== Fusion Module ====================
        self.FUSION_N_LAYERS = 2
        self.FUSION_N_HEADS = 8
        self.SENT_COND_MODE = 'none'  # 'none', 'prefix', 'kv_pool'

        # ==================== Phonological Attributes ====================
        self.USE_PHONO = False
        self.PHONO_DIM = 64
        self.SIGNBANK_CSV = 'data/ASL_signbank/asl_signbank_dictionary-export.csv'



class Phoenix2D_Config(BaseConfig):
    """RWTH-PHOENIX-2014T 2D-keypoint training config."""

    def __init__(self):
        super().__init__()
        self.DATASET_NAME      = "Phoenix2D"
        self.PHOENIX_ROOT      = "/scratch/rhong5/dataset/RWTH-PHOENIX-2014"
        self.PHOENIX_USE_FACE  = False
        # body 33 + lhand 21 + rhand 21 = 75 keypoints * 3 dims = 225
        self.MAX_SEQ_LEN       = 200
        self.TARGET_SEQ_LEN    = 160
        self.INTERPOLATE_SHORT_SEQ = False

        # Phoenix has no SMPL-X joint structure → treat input as flat features.
        self.USE_3D_INPUT      = True
        self.ROOT_NORMALIZE    = True   # dataset centres on hip; model bypass off via USE_3D_INPUT
        self.USE_UPPER_BODY    = True   # drop MediaPipe pose 25..32 (legs/feet)

        # Loss split inside CFG trainer when DATASET_NAME=Phoenix2D.
        # body_k=25 when USE_UPPER_BODY; sync via cfg.USE_UPPER_BODY at training time.
        self.PHOENIX_BODY_K    = 25     # head + arms + hips
        self.PHOENIX_HAND_K    = 21

        self.LATENT_DIM        = 256
        self.MODEL_DIM         = 512
        self.N_HEADS           = 8
        self.N_LAYERS          = 4
        self.DROPOUT           = 0.1

        self.TRAIN_BSZ         = 64
        self.EVAL_BSZ          = 64
        self.MAX_EPOCHS        = 200
        self.LEARNING_RATE     = 1e-4

        self.MODEL_VERSION     = 'v1'
        self.DATASET_VERSION   = 'v1'

        # Phoenix gloss source for cond_mode='gloss'
        self.GLOSS_SOURCE      = 'gt'   # 'gt' (orth) | 'translation'


class How2Sign_SMPLX_Config(BaseConfig):
    """Training configuration"""

    def __init__(self):
        super().__init__()
        # ==================== Dataset ====================
        
        # ==================== ASLLVD_Skeleton3D ====================
        self.DATASET_NAME = "How2SignSMPLX"
        self.ROOT_DIR ='/scratch/rhong5/dataset/Neural-Sign-Actors'
        # self.ROOT_DIR ='/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/data/synthetic_smplx_data'
        

        # ==================== Data Dimensions ====================

        # self.INPUT_DIM = 159 # for dataset v1
        # self.INPUT_DIM = 264 # for dataset v2
        
        self.MAX_SEQ_LEN = 200
        
        # Sequence interpolation (original data has only 2-4 frames per sample)
        self.TARGET_SEQ_LEN = 200  # Target sampling sequence length
        self.INTERPOLATE_SHORT_SEQ = False  # Whether to interpolate short sequences
                
        # ==================== Model Architecture ====================
        self.LATENT_DIM = 256
        self.MODEL_DIM = 512
        self.N_HEADS = 8
        self.N_LAYERS = 4
        self.DROPOUT = 0.1
        
        # ==================== Training ====================
        self.TRAIN_BSZ = 100
        self.EVAL_BSZ = 100
        
        self.MODEL_VERSION = 'v2'
        self.DATASET_VERSION = 'v1'

        self.USE_ROT6D = True
        self.USE_UPPER_BODY = True
        