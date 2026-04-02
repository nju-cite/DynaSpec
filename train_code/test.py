import sys
import os
import time
import random
import datetime
import argparse
from collections import OrderedDict

import torch
import numpy as np
import scipy.io as scio

from architecture import model_generator
from option import parse_opt
from utils import *
from log import *


# ---- Parse CLI arguments ----
parser_select = argparse.ArgumentParser(description="PGDSRT Testing")
parser_select.add_argument('--test_device', type=int, default=0, help="GPU device for testing")
parser_select.add_argument('--model_test_path', type=str, default=None,
                           help="override model path for testing")
args_select, unknown = parser_select.parse_known_args()

# Pass remaining args to option parser
sys.argv = [sys.argv[0]] + unknown
opt = parse_opt()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if not os.getenv('CUDA_VISIBLE_DEVICES', None):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
opt.num_gpu = torch.cuda.device_count()
opt.batch_size = opt.batch_size_per_gpu * opt.num_gpu


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if not torch.cuda.is_available():
    raise Exception('NO GPU!')

seed = opt.manual_seed
if seed is None:
    seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
set_seed(seed)


# ---- Test crop parameters (hardcoded) ----
# These define the crop positions for the 15 test scenes (10 static + 5 dynamic)
# Format: each list has 15 entries corresponding to 15 test scenes
test_crop_params = {
    'test_h_step': [
        np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3),
        np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3),
        np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3),
    ],
    'test_w_step': [
        np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3),
        np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3),
        np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3), np.array([0]*3),
    ],
    'test_flip': [
        (False, False), (False, False), (False, False), (False, False), (False, False),
        (False, False), (False, False), (False, False), (False, False), (False, False),
        (False, False), (False, False), (False, False), (False, False), (False, False),
    ],
    'test_h_starts': [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
    'test_w_starts': [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
}

# ---- Dataset ----
test_data_class = LoadTest_dyna(opt.test_path_static, opt.test_path_dyna)

# ---- Masks (sequence test) ----
dyna_sequence = 10
process_times = len(list(range(0, dyna_sequence - opt.frame_num_test, opt.frame_num_test)))
mask3d_batch_test, input_mask_test = init_mask_video(
    len(test_data_class['imgs_static']) + len(test_data_class['imgs_dyna']) * process_times, opt, mode='test')

test_gt = prepare_test_gt(test_data_class, opt, test_crop_params, mode='test').float()
b, f, c, h, w = test_gt.shape
input_meas_test = init_meas(opt, test_gt.reshape(b * f, c, h, w), mask3d_batch_test, mean_meas=opt.mean_meas)

# ---- Saving path ----
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path_root = opt.outf + date_time
if opt.tag != '':
    result_path_root = opt.outf[:-1] + '_' + opt.tag + '/' + date_time
if opt.tag_infolder != '':
    result_path_root = result_path_root + '_' + opt.tag_infolder
result_path_root = result_path_root.replace('exp', 'exp_test', 1)
result_path = result_path_root + '/result/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# ---- Model ----
model = model_generator(opt, mode='test', model_load_path=args_select.model_test_path)
model.cuda()


def test(model, test_gt, input_meas_test, mask3d_batch_test, input_mask_test, logger):
    torch.cuda.empty_cache()
    device_select = 'cuda:' + str(args_select.test_device)
    device_store = 'cpu'
    model.eval()
    model = model.to(device_select)
    psnr_list, ssim_list, sam_list = [], [], []
    test_gt = test_gt.to(device_store)
    b, f, c, h, w = test_gt.shape

    if input_mask_test is not None:
        if isinstance(input_mask_test, tuple):
            (Phi_batch_temp, Phi_s_batch_temp) = input_mask_test
            Phi_batch_temp = Phi_batch_temp.reshape([b, f] + list(Phi_batch_temp.shape[1:])).to(device_select)
            Phi_s_batch_temp = Phi_s_batch_temp.reshape([b, f] + list(Phi_s_batch_temp.shape[1:])).to(device_select)
            input_mask_test_class = (Phi_batch_temp, Phi_s_batch_temp)
        else:
            input_mask_test_class = input_mask_test.reshape([b, f] + list(input_mask_test.shape[1:])).to(device_select)
        mask3d_batch_test = mask3d_batch_test.to(device_select)

    input_meas_test = input_meas_test.reshape([b, f] + list(input_meas_test.shape[1:])).to(device_select)

    reconstruct_result = torch.zeros_like(test_gt)
    with torch.no_grad():
        time_per_sceneframe = []
        for scene_idx in range(test_gt.shape[0]):
            print('scene_idx:{}/{}'.format(scene_idx + 1, test_gt.shape[0]), flush=True)
            input_meas_temp = input_meas_test[scene_idx]
            if input_mask_test is not None:
                if isinstance(input_mask_test_class, tuple):
                    (Phi_batch_temp, Phi_s_batch_temp) = input_mask_test_class
                    input_mask_test_temp = (Phi_batch_temp[scene_idx], Phi_s_batch_temp[scene_idx])
                    save_Phi_batch = Phi_batch_temp.clone()
                else:
                    input_mask_test_temp = input_mask_test_class[scene_idx]
                    save_Phi_batch = input_mask_test_class.clone()
            else:
                input_mask_test_temp = None
                save_Phi_batch = None

            begin_testtime_temp = time.time()
            model_out = model(input_meas_temp, input_mask_test_temp)
            time_per_sceneframe.append((time.time() - begin_testtime_temp) / f * 1000)
            reconstruct_result[scene_idx, ...] = model_out.to(device_store)

        flops_G, n_param = my_summary_test(model, H=256, W=256, C=28, N=1,
                                           input=input_meas_temp, mask=input_mask_test_temp)

    for k in range(test_gt.shape[0]):
        psnr_list_per, ssim_list_per, sam_list_per = [], [], []
        for frame_idx in range(test_gt.shape[1]):
            psnr_val = torch_psnr(reconstruct_result[k, frame_idx], test_gt[k, frame_idx])
            ssim_val = torch_ssim(reconstruct_result[k, frame_idx], test_gt[k, frame_idx])
            sam_val = torch_sam(reconstruct_result[k, frame_idx], test_gt[k, frame_idx])
            psnr_list_per.append(psnr_val.detach().cpu().numpy())
            ssim_list_per.append(ssim_val.detach().cpu().numpy())
            sam_list_per.append(sam_val.detach().cpu().numpy())
        psnr_list.append(psnr_list_per)
        ssim_list.append(ssim_list_per)
        sam_list.append(sam_list_per)

    pred = np.transpose(reconstruct_result.detach().cpu().numpy(), (0, 1, 3, 4, 2)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 1, 3, 4, 2)).astype(np.float32)
    input_meas_np = input_meas_test.cpu().numpy().astype(np.float32)
    if save_Phi_batch is not None:
        save_Phi_batch = save_Phi_batch.cpu().numpy().astype(np.float32)
    else:
        save_Phi_batch = np.zeros((1,), dtype=np.float32)

    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    sam_mean = np.mean(np.asarray(sam_list))
    per_frame_time = np.mean(np.asarray(time_per_sceneframe))

    logger.info('testing ave_psnr = {:.2f}, ave_ssim = {:.4f}, ave_sam = {:.4f}, per_frame_time: {:.2f} ms'.format(
        psnr_mean, ssim_mean, sam_mean, per_frame_time))

    torch.cuda.empty_cache()
    return pred, truth, psnr_list, ssim_list, sam_list, psnr_mean, ssim_mean, sam_mean, \
           time_per_sceneframe, per_frame_time, input_meas_np, save_Phi_batch, flops_G, n_param


def main():
    logger = gen_log(result_path)
    logger.info('Configuration:\n')
    logger.info(dict2str(OrderedDict(vars(opt))))
    logger.info(tcolor("----------- Imaging System -----------", c=Color.Yellow))
    logger.info(tcolor("Dispersion:  ", c=Color.Yellow) + opt.dispersion)
    logger.info(tcolor("Mask Pattern:  ", c=Color.Yellow) + opt.mask_pattern)
    logger.info(tcolor("Spectral Bands:  ", c=Color.Yellow) + str(opt.SpectralBands))
    logger.info(tcolor("Disperse Step:  ", c=Color.Yellow) + str(opt.step))
    logger.info(tcolor("Save Folder:  ", c=Color.Yellow) + str(result_path_root))
    logger.info('')

    set_seed(seed)
    model.eval()

    (pred, truth, psnr_all, ssim_all, sam_all,
     psnr_mean, ssim_mean, sam_mean,
     time_per_sceneframe, per_frame_time,
     input_meas, save_Phi_batch, flops_G, n_param) = test(
        model, test_gt, input_meas_test, mask3d_batch_test, input_mask_test, logger)

    test_eval_path = result_path + "/E_PSNR_{:.2f}.txt".format(psnr_mean)
    write_results_to_txt(psnr_all, ssim_all, sam_all, time_per_sceneframe,
                         output_file=test_eval_path, distinguish_static=True,
                         model_property=(flops_G, n_param), frame_num=opt.frame_num_test)

    name = result_path + '/Test_{:.2f}_{:.3f}.mat'.format(psnr_mean, ssim_mean)
    scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all,
                        'sam_list': sam_all, 'input_meas': input_meas, 'save_Phi_batch': save_Phi_batch},
                 do_compression=False, format='5')


if __name__ == '__main__':
    main()
