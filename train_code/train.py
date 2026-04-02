import sys
import os
import time
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
parser_select = argparse.ArgumentParser(description="PGDSRT Training")
parser_select.add_argument('--test_device', type=int, default=0, help="GPU device for testing")
parser_select.add_argument('--begin_epoch', type=int, default=1, help="begin epoch (for resuming)")
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
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)


if not torch.cuda.is_available():
    raise Exception('NO GPU!')

seed = opt.manual_seed
if seed is None:
    seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
set_seed(seed)

begin_epoch = args_select.begin_epoch

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
train_set = LoadTraining_dyna(opt.data_path_static, opt.data_path_dyna)
test_data_class = LoadTest_dyna(opt.test_path_static, opt.test_path_dyna)

# ---- Masks ----
mask3d_batch_train, input_mask_train = init_mask_video(opt.batch_size, opt)
if input_mask_train is not None:
    if isinstance(input_mask_train, tuple):
        (Phi_batch, Phi_s_batch) = input_mask_train
        Phi_batch = Phi_batch.cuda()
        Phi_s_batch = Phi_s_batch.cuda()
        input_mask_train = (Phi_batch, Phi_s_batch)
    else:
        input_mask_train = input_mask_train.cuda()
mask3d_batch_train = mask3d_batch_train.cuda()

mask3d_batch_test, input_mask_test = init_mask_video(
    len(test_data_class['imgs_static']) + len(test_data_class['imgs_dyna']), opt, mode='test')

# ---- Test GT ----
test_gt = prepare_test_gt(test_data_class, opt, test_crop_params).float()
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
result_path = result_path_root + '/result/'
model_path = result_path_root + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# ---- Model ----
model = model_generator(opt)
model = torch.nn.DataParallel(model).cuda()

# ---- Optimizer ----
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=0)

# ---- Scheduler ----
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)

for _ in range(1, begin_epoch):
    scheduler.step()

# ---- Loss ----
if opt.lossfn_type == 'l1':
    lossfn = torch.nn.L1Loss().cuda()
elif opt.lossfn_type == 'l2':
    lossfn = torch.nn.MSELoss().cuda()
elif opt.lossfn_type == 'rmse':
    lossfn = lambda x, y: (torch.nn.MSELoss()(x, y) + 1e-8).sqrt()
else:
    raise NotImplementedError('Loss type [{:s}] is not found.'.format(opt.lossfn_type))

training_start_time = time.time()


def _format_eta(seconds):
    """Format seconds into human-readable ETA string (e.g. '1d 03:25:10')."""
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return '{}d {:02d}:{:02d}:{:02d}'.format(days, hours, minutes, secs)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, secs)


def train(epoch, logger):
    epoch_loss = 0
    epoch_psnr = 0
    epoch_ssim = 0
    epoch_sam = 0
    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))

    for i in range(batch_num):
        gt_batch = shuffle_crop_dyna_random(train_set, opt.batch_size, opt.frame_num, opt)
        gt = gt_batch.cuda().float()
        b, f, c, h, w = gt.shape
        gt = gt.view(b * f, c, h, w)
        input_meas = init_meas(opt, gt, mask3d_batch_train, mean_meas=opt.mean_meas)

        optimizer.zero_grad()
        model_out = model(input_meas, input_mask_train)
        loss = lossfn(model_out, gt)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        if (i + 1) % opt.print_info == 0:
            reconstruct_result = model_out.reshape(b * f * c, h, w).cpu()
            gt_temp = gt.reshape(b * f * c, h, w).cpu()
            psnr_train = torch_psnr(reconstruct_result, gt_temp)
            ssim_train = torch_ssim(reconstruct_result, gt_temp)
            sam_train = torch_sam(reconstruct_result, gt_temp)
            epoch_psnr += psnr_train
            epoch_ssim += ssim_train
            epoch_sam += sam_train

            progress = (i + 1) / batch_num
            bar_len = 20
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            current_step = (epoch - 1) * batch_num + i + 1
            est_remaining = (time.time() - training_start_time) / current_step * (
                        opt.max_epoch * batch_num - current_step)

            logger.info(
                tcolor('[{bar}] ', c=Color.Cyan).format(bar=bar)
                + 'Epoch {epoch}/{max_epoch} [{batch}/{total}]  '
                  'PSNR: {psnr:.4f}  SSIM: {ssim:.4f}  SAM: {sam:.4f}  '
                  'ETA: {eta}'.format(
                    epoch=epoch, max_epoch=opt.max_epoch,
                    batch=i + 1, total=batch_num,
                    psnr=psnr_train, ssim=ssim_train, sam=sam_train,
                    eta=_format_eta(est_remaining)))

    end = time.time()
    n_prints = batch_num // opt.print_info
    n_prints = max(n_prints, 1)
    logger.info(
        tcolor("── Epoch {} Summary ──", c=Color.Green).format(epoch)
        + '  Loss: {:.6f}  PSNR: {:.4f}  SSIM: {:.4f}  SAM: {:.4f}  Time: {:.1f}min'.format(
            epoch_loss / batch_num,
            epoch_psnr / n_prints,
            epoch_ssim / n_prints,
            epoch_sam / n_prints,
            (end - begin) / 60))
    return 0


def test(epoch, model, test_gt, input_meas_test, mask3d_batch_test, input_mask_test, logger):
    torch.cuda.empty_cache()
    device_store = 'cpu'
    device_select = 'cuda:' + str(args_select.test_device)
    model.eval()
    model = model.module
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
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    return pred, truth, psnr_list, ssim_list, sam_list, psnr_mean, ssim_mean, sam_mean, \
           time_per_sceneframe, per_frame_time, input_meas_np, save_Phi_batch


def main():
    logger = gen_log(model_path)
    logger.info('Configuration:\n')
    logger.info(dict2str(OrderedDict(vars(opt))))
    logger.info(tcolor("----------- Imaging System -----------", c=Color.Yellow))
    logger.info(tcolor("Dispersion:  ", c=Color.Yellow) + opt.dispersion)
    logger.info(tcolor("Mask Pattern:  ", c=Color.Yellow) + opt.mask_pattern)
    logger.info(tcolor("Spectral Bands:  ", c=Color.Yellow) + str(opt.SpectralBands))
    logger.info(tcolor("Disperse Step:  ", c=Color.Yellow) + str(opt.step))
    logger.info(tcolor("----------- Training -----------", c=Color.Yellow))
    logger.info(tcolor("Max Epochs:  ", c=Color.Yellow) + str(opt.max_epoch))
    logger.info(tcolor("Learning Rate:  ", c=Color.Yellow) + str(opt.learning_rate))
    logger.info(tcolor("Batch Size:  ", c=Color.Yellow) + str(opt.batch_size))
    logger.info(tcolor("Frame Num:  ", c=Color.Yellow) + str(opt.frame_num))
    logger.info(tcolor("----------- Output Paths -----------", c=Color.Yellow))
    logger.info(tcolor("Experiment Root:  ", c=Color.Yellow) + str(result_path_root))
    logger.info(tcolor("Checkpoints:      ", c=Color.Yellow) + str(model_path))
    logger.info(tcolor("Test Results:     ", c=Color.Yellow) + str(result_path))
    logger.info(tcolor("Training Log:     ", c=Color.Yellow) + str(model_path) + 'log.txt')
    logger.info(tcolor("Evaluation:       ", c=Color.Yellow) + str(result_path_root) + '/evaluation.txt')
    logger.info('')

    (psnr_max, corres_ssim, corres_sam, corres_epoch) = 0, 0, 0, 0
    set_seed(seed)

    for epoch in range(begin_epoch, opt.max_epoch + 1):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(tcolor('===> Epoch {}/{}'.format(epoch, opt.max_epoch), c=Color.Blue, m=Mode.Background)
                    + '  LR: {:.2e}'.format(current_lr))

        model.train()
        train(epoch, logger)
        model.eval()

        (pred, truth, psnr_all, ssim_all, sam_all,
         psnr_mean, ssim_mean, sam_mean,
         time_per_sceneframe, per_frame_time,
         input_meas, save_Phi_batch) = test(
            epoch, model, test_gt, input_meas_test, mask3d_batch_test, input_mask_test, logger)

        scheduler.step()

        test_evaluation_path = result_path_root + "/evaluation.txt"
        write_results_to_txt(psnr_all, ssim_all, sam_all, time_per_sceneframe,
                             output_file=test_evaluation_path, distinguish_static=True)

        if epoch < opt.max_epoch // 2:
            psnr_thre = 37

        if (epoch > opt.max_epoch // 2) and (psnr_max < psnr_thre):
            psnr_thre = psnr_max

        if psnr_mean > psnr_max:
            (psnr_max, corres_ssim, corres_sam, corres_epoch) = psnr_mean, ssim_mean, sam_mean, epoch
            if psnr_mean > psnr_thre:
                checkpoint(model, epoch, model_path, logger)
                test_eval_path = model_path + "/E_epoch_{}_PSNR_{:.2f}.txt".format(epoch, psnr_mean)
                write_results_to_txt(psnr_all, ssim_all, sam_all, time_per_sceneframe,
                                     output_file=test_eval_path, distinguish_static=True)

        if epoch > (opt.max_epoch - 2):
            checkpoint(model, epoch, model_path, logger)
            name = result_path + '/Test_{}_{:.2f}_{:.3f}.mat'.format(epoch, psnr_max, ssim_mean)
            scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all,
                                'sam_list': sam_all, 'input_meas': input_meas, 'save_Phi_batch': save_Phi_batch},
                         do_compression=False, format='5')

        logger.info(tcolor("★ Best", c=Color.Magenta) +
                    " @ Epoch {}: PSNR=".format(corres_epoch) + gradient_num_color(psnr_max) +
                    '  SSIM={:.4f}  SAM={:.4f}\n'.format(corres_ssim, corres_sam))


if __name__ == '__main__':
    main()
