import argparse


def parse_opt():
    """Parse command-line options for PGDSRT training and evaluation.

    Two key parameters define the imaging system configuration:
      --dispersion:   'single' or 'dual'
      --mask_pattern: 'random', 'notch', or 'pmvis'

    The four valid imaging system combinations are:

    Imaging System       | dispersion | mask_pattern | Forward Model
    ---------------------|------------|--------------|----------------
    SD-CASSI             | single     | random       | PGDSRT_sd
    PMVIS                | single     | pmvis        | PGDSRT_sd
    DD-CASSI (default)   | dual       | random       | PGDSRT_dd
    NDSSI                | dual       | notch        | PGDSRT_dd

    Other combinations (e.g. single+notch, dual+pmvis) are rejected at parse time.
    """
    parser = argparse.ArgumentParser(
        description="PGDSRT: Dynamic Spectral Image Reconstruction"
    )

    # ---- Hardware ----
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="CUDA visible device id(s), e.g. '0' or '0,1'")
    parser.add_argument("--manual_seed", type=int, default=317,
                        help="random seed for reproducibility")

    # ---- Data paths ----
    parser.add_argument("--data_root", type=str,
                        default='./../../../../cljdata/Dynaspec/',
                        help="root directory for training / test datasets")
    parser.add_argument("--mask_root", type=str,
                        default='./../mask_folder',
                        help="directory containing coded-aperture masks")

    # ---- Imaging system ----
    # Valid combinations:
    #   single + random  → SD-CASSI
    #   single + pmvis   → PMVIS
    #   dual   + random  → DD-CASSI (default)
    #   dual   + notch   → NDSSI
    parser.add_argument("--dispersion", type=str, default='dual',
                        choices=['single', 'dual'],
                        help="dispersion type: 'single' (SD-CASSI/PMVIS) "
                             "or 'dual' (DD-CASSI/NDSSI)")
    parser.add_argument("--mask_pattern", type=str, default='random',
                        choices=['random', 'notch', 'pmvis'],
                        help="mask pattern: 'random', 'notch', or 'pmvis'")
    parser.add_argument("--SpectralBands", type=int, default=30,
                        help="number of spectral bands to reconstruct")
    parser.add_argument("--step", type=int, default=2,
                        help="dispersive shift step (pixels per band)")
    parser.add_argument("--frame_num", type=int, default=3,
                        help="number of temporal frames for training")
    parser.add_argument("--frame_num_test", type=int, default=3,
                        help="number of temporal frames for testing")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="spatial crop size for training patches")
    parser.add_argument("--mask_size", type=int, default=256,
                        help="coded-aperture mask spatial size")
    parser.add_argument("--static_probability", type=float, default=0.3,
                        help="probability of sampling a static scene")
    parser.add_argument("--dyna_not_move_probability", type=float, default=0.3,
                        help="probability that a dynamic scene has no motion")
    parser.add_argument("--move_step", type=int, nargs=2, default=[0, 10],
                        help="min and max pixel shift for dynamic augmentation")

    # ---- Training ----
    parser.add_argument("--batch_size_per_gpu", type=int, default=1,
                        help="batch size per GPU")
    parser.add_argument("--max_epoch", type=int, default=80,
                        help="total training epochs")
    parser.add_argument("--epoch_sam_num", type=int, default=2000,
                        help="number of samples drawn per epoch")
    parser.add_argument("--learning_rate", type=float, default=0.0003,
                        help="initial learning rate")
    parser.add_argument("--optimizer_type", type=str, default='adam',
                        help="optimizer: 'adam'")
    parser.add_argument("--lossfn_type", type=str, default='rmse',
                        help="loss function: 'l1', 'l2', 'rmse', 'ssim', etc.")
    parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR',
                        help="LR scheduler: 'MultiStepLR' or 'CosineAnnealingLR'")
    parser.add_argument("--milestones", type=int, nargs='+',
                        default=[14, 28, 42, 56, 70],
                        help="epoch milestones for MultiStepLR")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="LR decay factor at each milestone")
    parser.add_argument("--print_info", type=int, default=200,
                        help="print training info every N iterations")

    # ---- Model ----
    parser.add_argument("--method", type=str, default='pgdsrt',
                        help="reconstruction method (fixed: pgdsrt)")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="path to a pretrained checkpoint to resume from")
    parser.add_argument("--simexp_model_path", type=str,
                        default='./../model_zoo/pgdsrt/model_epoch_<xx>.pth',
                        help="path to the simulation-trained model for eval")

    # ---- Output ----
    parser.add_argument("--outf_root", type=str, default='./exp/',
                        help="root directory for experiment outputs")
    parser.add_argument("--tag", type=str, default='',
                        help="optional tag appended to experiment name")
    parser.add_argument("--tag_infolder", type=str, default='',
                        help="optional sub-folder tag inside experiment dir")

    opt = parser.parse_args()

    # ---- Validate imaging system combination ----
    _valid_combos = {
        ('single', 'random'),   # SD-CASSI
        ('single', 'pmvis'),    # PMVIS
        ('dual',   'random'),   # DD-CASSI
        ('dual',   'notch'),    # NDSSI
    }
    if (opt.dispersion, opt.mask_pattern) not in _valid_combos:
        parser.error(
            f"Invalid combination: --dispersion={opt.dispersion} --mask_pattern={opt.mask_pattern}\n"
            f"  Valid: single+random (SD-CASSI), single+pmvis (PMVIS), "
            f"dual+random (DD-CASSI), dual+notch (NDSSI)"
        )

    # ---- Derived data paths ----
    opt.mask_path = f"{opt.mask_root}/"
    opt.data_path_static = f"{opt.data_root}/CAVE_512_50030/"
    opt.data_path_dyna = f"{opt.data_root}/Dyna_Spec_train_320_ref_norm_50030_train/"
    opt.test_path_static = f"{opt.data_root}/Kaist_Truth_50030/"
    opt.test_path_dyna = f"{opt.data_root}/Dyna_Spec_train_320_ref_norm_50030_test/"
    opt.outf = f"{opt.outf_root}/pgdsrt/"

    # ---- Fixed network settings (hardcoded in the model) ----
    opt.input_setting = 'Y'            # network input is raw measurement Y; model computes H internally
    opt.input_mask = 'Phi_PhiPhiT'     # network receives Phi and sum(Phi^2)
    opt.mean_meas = False              # model normalizes (meas / nC * 2) internally

    # ---- Backward compatibility ----
    opt.EncodingArch = 'sd-cassi' if opt.dispersion == 'single' else 'dd-cassi'

    return opt
