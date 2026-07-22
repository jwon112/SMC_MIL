import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from huggingface_hub import hf_hub_download

# UNI 체크포인트 저장 경로
UNI_CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'ckpts', 'uni')
UNI2_H_CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'ckpts', 'uni2_h')
UNI2_L_CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'ckpts', 'uni2_l')
CONCH_V1_5_CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'ckpts', 'conch_v1_5')

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def get_UNI_ckpt_path():
    """UNI v1 체크포인트 경로 반환. 없으면 HuggingFace에서 자동 다운로드."""
    # 환경변수가 설정되어 있으면 우선 사용
    if 'UNI_CKPT_PATH' in os.environ:
        return os.environ['UNI_CKPT_PATH']
    
    # 없으면 자동 다운로드
    os.makedirs(UNI_CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(UNI_CKPT_DIR, 'pytorch_model.bin')
    
    if not os.path.exists(ckpt_path):
        print('UNI v1 checkpoint not found. Downloading from HuggingFace...')
        hf_hub_download(
            repo_id="MahmoodLab/UNI",
            filename="pytorch_model.bin",
            local_dir=UNI_CKPT_DIR
        )
        print(f'UNI v1 checkpoint downloaded to {ckpt_path}')
    
    return ckpt_path

def get_UNI2_ckpt_path(variant='h'):
    """UNI v2 체크포인트 경로 반환. 없으면 HuggingFace에서 자동 다운로드.
    
    Args:
        variant: 'h' for ViT-H (larger, better), 'l' for ViT-L (same size as v1)
    """
    if variant == 'h':
        ckpt_dir = UNI2_H_CKPT_DIR
        repo_id = "MahmoodLab/UNI2-h"
        env_var = 'UNI2_H_CKPT_PATH'
    else:
        ckpt_dir = UNI2_L_CKPT_DIR
        repo_id = "MahmoodLab/UNI2-l"
        env_var = 'UNI2_L_CKPT_PATH'
    
    # 환경변수가 설정되어 있으면 우선 사용
    if env_var in os.environ:
        return os.environ[env_var]
    
    # 없으면 자동 다운로드
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'pytorch_model.bin')
    
    if not os.path.exists(ckpt_path):
        print(f'UNI v2-{variant} checkpoint not found. Downloading from HuggingFace...')
        hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
            local_dir=ckpt_dir
        )
        print(f'UNI v2-{variant} checkpoint downloaded to {ckpt_path}')
    
    return ckpt_path

def get_CONCH_v1_5_ckpt_path():
    """CONCH v1.5 체크포인트 경로 반환. 없으면 HuggingFace에서 자동 다운로드."""
    # 환경변수가 설정되어 있으면 우선 사용
    if 'CONCH_V1_5_CKPT_PATH' in os.environ:
        return os.environ['CONCH_V1_5_CKPT_PATH']
    
    # 없으면 자동 다운로드
    os.makedirs(CONCH_V1_5_CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CONCH_V1_5_CKPT_DIR, 'pytorch_model_vision.bin')
    
    if not os.path.exists(ckpt_path):
        print('CONCH v1.5 checkpoint not found. Downloading from HuggingFace...')
        hf_hub_download(
            repo_id="MahmoodLab/conchv1_5",
            filename="pytorch_model_vision.bin",
            local_dir=CONCH_V1_5_CKPT_DIR
        )
        print(f'CONCH v1.5 checkpoint downloaded to {ckpt_path}')
    
    return ckpt_path
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        uni_ckpt_path = get_UNI_ckpt_path()
        model = timm.create_model("vit_large_patch16_224",
                            img_size=224,
                            patch_size=16,
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(uni_ckpt_path, map_location="cpu"), strict=True)
    elif model_name == 'uni_v2':
        # Official UNI2-h architecture (ViT-h/14-reg8, 1536-dimensional output).
        uni2_ckpt_path = get_UNI2_ckpt_path(variant='h')
        model = timm.create_model(
            "vit_giant_patch14_224",
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load(uni2_ckpt_path, map_location="cpu"), strict=True)
    elif model_name == 'uni_v2_l':
        # UNI v2 ViT-L (same size as v1, faster)
        uni2_ckpt_path = get_UNI2_ckpt_path(variant='l')
        model = timm.create_model("vit_large_patch16_224",
                            img_size=224,
                            patch_size=16,
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(uni2_ckpt_path, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        # CONCH v1.5는 ViT-L/16 모델, timm으로 직접 로드
        conch_v1_5_ckpt_path = get_CONCH_v1_5_ckpt_path()
        model = timm.create_model("vit_large_patch16_224",
                            img_size=448,
                            patch_size=16,
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        # CONCH v1.5 체크포인트는 'trunk.' 접두사가 있으므로 제거
        checkpoint = torch.load(conch_v1_5_ckpt_path, map_location="cpu")
        # 'trunk.' 접두사 제거 및 vision 관련 키만 필터링
        state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('trunk.'):
                new_key = key[6:]  # 'trunk.' 제거 (6글자)
                state_dict[new_key] = value
        model.load_state_dict(state_dict, strict=True)
        assert target_img_size == 448, 'CONCH v1.5 is used with 448x448 input size'
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms
