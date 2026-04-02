import torch

from .PGDSRT import PGDSRT_dd, PGDSRT_sd


def model_generator(opt, mode='train', model_load_path=None):
    """Create and optionally load a PGDSRT model.

    Selects the single- or dual-dispersion variant based on opt.dispersion,
    and loads pretrained weights when available.

    Args:
        opt: Configuration namespace. Key fields:
            - dispersion (str): 'single' or 'dual'.
            - SpectralBands, crop_size, step, frame_num: Model architecture params.
            - pretrained_model_path (str): Checkpoint path for training resume.
            - simexp_model_path (str): Checkpoint path for evaluation.
        mode (str): 'train' or 'test'.
        model_load_path (str or None): Explicit checkpoint path (overrides opt paths).

    Returns:
        torch.nn.Module: Initialized model on CUDA.
    """
    # Select model variant based on dispersion type
    if opt.dispersion == 'single':
        model = PGDSRT_sd(
            bands=opt.SpectralBands,
            size=opt.crop_size,
            step=opt.step,
            frame=opt.frame_num,
            numblock=[2, 4, 4],
        ).cuda()
    elif opt.dispersion == 'dual':
        model = PGDSRT_dd(
            bands=opt.SpectralBands,
            size=opt.crop_size,
            step=opt.step,
            frame=opt.frame_num,
            numblock=[2, 4, 4],
        ).cuda()
    else:
        raise ValueError(f"Unknown dispersion type: {opt.dispersion}. Expected 'single' or 'dual'.")

    # Determine model load path based on mode
    if mode == 'train':
        if opt.pretrained_model_path is not None:
            model_load_path = opt.pretrained_model_path
    elif mode == 'test':
        if model_load_path is None or model_load_path == '':
            model_load_path = opt.simexp_model_path
        if model_load_path is None or model_load_path == '':
            print(f'Model path is not defined!')
            model_load_path = None

    # Load model weights if a path is provided
    if model_load_path is not None:
        print(f'Loading model from {model_load_path}')
        checkpoint = torch.load(model_load_path, map_location='cuda:0', weights_only=True)
        # Strip 'module.' prefix from keys (saved by DataParallel)
        checkpoint_clean = {}
        for key, value in checkpoint.items():
            if key.startswith('module.'):
                checkpoint_clean[key[7:]] = value
            else:
                checkpoint_clean = checkpoint
                break
        model.load_state_dict(checkpoint_clean, strict=True)

    return model
