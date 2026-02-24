


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
        self.MAX_EPOCHS = 500
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
        
        # ==================== Logging & Checkpoints ====================
        self.PROJECT_NAME = "ASLAvatar_V1"
        self.LOG_INTERVAL = 50
        self.EVAL_INTERVAL = 5  # Evaluate every N epochs

        
        # Paths (adjust to your environment)
        self.LOG_DIR = "/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/zlog"
        self.CKPT_DIR = "/scratch/rhong5/weights/temp_training_weights/"
        
        
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

class ASLLVD_Skeleton3D_Config(BaseConfig):
    """Training configuration"""
    
    def __init__(self):
        super().__init__()
        # ==================== Dataset ====================
        
        # ==================== ASLLVD_Skeleton3D ====================
        self.DATASET_NAME = "ASLLVD_Skeleton3D"
        self.SKELETON_DIR ='/scratch/rhong5/dataset/ASLLVD/asl-skeleton3d/normalized/3d'
        self.PHONO_DIR = '/scratch/rhong5/dataset/ASLLVD/asl-phono/phonology/3d'
        
        # Split files (auto-generated if not exist)
        self.TRAIN_SPLIT_FILE = "/scratch/rhong5/dataset/ASLLVD/train_split.txt"
        self.TEST_SPLIT_FILE = "/scratch/rhong5/dataset/ASLLVD/test_split.txt"
        
        # ==================== Data Dimensions ====================
        # Upper Body(14) + Face(16) + HandL(21) + HandR(21) = 72 joints
        # 72 * 3 (x,y,z) = 216
        self.INPUT_DIM = 216
        self.MAX_SEQ_LEN = 50
        
        # Sequence interpolation (original data has only 2-4 frames per sample)
        self.TARGET_SEQ_LEN = 8  # Target sampling sequence length
        self.INTERPOLATE_SHORT_SEQ = True  # Whether to interpolate short sequences
                
        # ==================== Training ====================
        self.TRAIN_BSZ = 400
        self.EVAL_BSZ = 200
        

    


class SignBank_SMPLX_Config(BaseConfig):
    """Training configuration"""
    
    def __init__(self):
        super().__init__()
        # ==================== Dataset ====================
        
        # ==================== ASLLVD_Skeleton3D ====================
        self.DATASET_NAME = "SignBank_SMPLX"
        self.ROOT_DIR ='/scratch/rhong5/dataset/asl_signbank/smplx_params'

        # ==================== Data Dimensions ====================

        self.INPUT_DIM = 159
        self.MAX_SEQ_LEN = 100
        
        # Sequence interpolation (original data has only 2-4 frames per sample)
        self.TARGET_SEQ_LEN = 40  # Target sampling sequence length
        self.INTERPOLATE_SHORT_SEQ = False  # Whether to interpolate short sequences
                
        # ==================== Model Architecture ====================
        self.LATENT_DIM = 256
        self.MODEL_DIM = 512
        self.N_HEADS = 8
        self.N_LAYERS = 4
        self.DROPOUT = 0.1
        
        # ==================== Training ====================
        self.TRAIN_BSZ = 400
        self.EVAL_BSZ = 200
        
        self.MODEL_VERSION = 'v5'
        self.DATASET_VERSION = 'v2'



class WLASL_SMPLX_Config(BaseConfig):
    """Training configuration"""
    
    def __init__(self):
        super().__init__()
        # ==================== Dataset ====================
        
        # ==================== ASLLVD_Skeleton3D ====================
        self.DATASET_NAME = "WLASL_SMPLX"
        self.ROOT_DIR ='/scratch/rhong5/dataset/wlasl'
        # self.ROOT_DIR ='/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/data/synthetic_smplx_data'
        

        # ==================== Data Dimensions ====================

        # self.INPUT_DIM = 159 # for dataset v1
        # self.INPUT_DIM = 264 # for dataset v2
        
        self.MAX_SEQ_LEN = 100
        
        # Sequence interpolation (original data has only 2-4 frames per sample)
        self.TARGET_SEQ_LEN = 40  # Target sampling sequence length
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
        
        self.MODEL_VERSION = 'v5'
        self.DATASET_VERSION = 'v2'

        self.USE_ROT6D = True
        self.USE_UPPER_BODY = True
        

