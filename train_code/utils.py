import scipy.io as sio
import os
import numpy as np
import torch
import random
import time
from ssim_torch import ssim
import torch.nn.functional as F
import h5py
import math
import cv2
from torch.utils.data import Dataset
from fvcore.nn import FlopCountAnalysis


# ============================================================================
# Data Loading
# ============================================================================

def LoadTraining_dyna(path_static, path_dyna):
    """Load training data (static scenes + dynamic video sequences).

    Args:
        path_static (str): Path to static scene .mat files (CAVE dataset).
        path_dyna (str): Path to dynamic scene directories, each containing
                         per-frame .mat files.

    Returns:
        dict: {'imgs_static': list[ndarray(H,W,C)], 'imgs_dyna': list[list[ndarray(H,W,C)]]}
              Static scenes are resized to (384, 384). Values normalized to [0, 1].
    """
    if path_static is not None:
        imgs_static = []
        static_scene_list = os.listdir(path_static)
        static_scene_list.sort()
        print('Static training scenes:', len(static_scene_list))
        for i in range(len(static_scene_list)):
            scene_path = path_static + static_scene_list[i]
            scene_num = int(static_scene_list[i].split('.')[0][5:])
            if scene_num <= 30:
                if 'mat' not in scene_path:
                    continue
                try:
                    img_dict = sio.loadmat(scene_path)
                    need_transpose = False
                except NotImplementedError:
                    img_dict = h5py.File(scene_path, 'r')
                    need_transpose = True
                except Exception as e:
                    print(f"Error loading {scene_path}: {e}")
                    raise e
                if "data_slice_31" in img_dict:
                    img = img_dict['data_slice_31'][:] / 65536.
                    if need_transpose:
                        img = img.transpose(2, 1, 0)
                    img = resize_images_bilateral(img, (384, 384))
                    img = np.clip(img, 0, 1)
                    img = img.astype(np.float32)
                    imgs_static.append(img)
                print('Static scene {}: {} is loaded.'.format(i, static_scene_list[i]))

    imgs_dyna = []
    dyna_scene_list = os.listdir(path_dyna)
    dyna_scene_list.sort()
    print('Dynamic training scenes:', len(dyna_scene_list))

    for i in range(len(dyna_scene_list)):
        dyna_dir_path = path_dyna + dyna_scene_list[i]
        dyna_scene_frame_list = os.listdir(dyna_dir_path)
        dyna_scene_frame_list.sort()
        imgs_dyna_perscene = []
        for f_idx in range(len(dyna_scene_frame_list)):
            sceneframe_path = dyna_dir_path + '/' + dyna_scene_frame_list[f_idx]
            if 'mat' not in sceneframe_path:
                continue
            try:
                img_dict = sio.loadmat(sceneframe_path)
                need_transpose = False
            except NotImplementedError:
                img_dict = h5py.File(sceneframe_path, 'r')
                need_transpose = True
            except Exception as e:
                print(f"Error loading {sceneframe_path}: {e}")
                raise e
            if "img" in img_dict:
                img = img_dict['img'][:] / 65536.
                if need_transpose:
                    img = img.transpose(2, 1, 0)
                img = np.clip(img, 0, 1)
                img = img.astype(np.float32)
                imgs_dyna_perscene.append(img)
                print('Dynamic scene {}: {}/{} is loaded.'.format(i, dyna_scene_list[i], dyna_scene_frame_list[f_idx]))
        imgs_dyna.append(imgs_dyna_perscene)

    if path_static is not None:
        imgs_class = {
            'imgs_static': imgs_static,
            'imgs_dyna': imgs_dyna
        }
    else:
        imgs_class = {
            'imgs_dyna': imgs_dyna
        }

    return imgs_class


def LoadTest_dyna(path_test_static, path_dyna):
    """Load test data (static scenes + dynamic video sequences).

    Args:
        path_test_static (str): Path to static test scene .mat files (Kaist dataset).
        path_dyna (str): Path to dynamic test scene directories.

    Returns:
        dict: {'imgs_static': list[ndarray(H,W,C)], 'imgs_dyna': list[list[ndarray(H,W,C)]]}
              Static scenes are resized to (320, 320). Values normalized to [0, 1].
    """
    if path_test_static is not None:
        scene_list_static = os.listdir(path_test_static)
        scene_list_static.sort()
        imgs_static = []

        for i in range(len(scene_list_static)):
            scene_path = path_test_static + scene_list_static[i]

            try:
                img_dict = sio.loadmat(scene_path)
                need_transpose = False
            except NotImplementedError:
                img_dict = h5py.File(scene_path, 'r')
                need_transpose = True
            except Exception as e:
                print(f"Error loading {scene_path}: {e}")
                raise e

            img = img_dict['img_31'][:]
            if need_transpose:
                img = img.transpose(2, 1, 0)
            img = resize_images_bilateral(img, (320, 320))
            img = np.clip(img, 0, 1)
            img = img.astype(np.float32)
            imgs_static.append(img)
            print('Test Static scene {}: {} is loaded.'.format(i, scene_list_static[i]))


    dyna_scene_list = os.listdir(path_dyna)
    dyna_scene_list.sort()
    imgs_dyna = []

    for i in range(len(dyna_scene_list)):
        dyna_dir_path = path_dyna + dyna_scene_list[i]
        dyna_scene_frame_list = os.listdir(dyna_dir_path)
        dyna_scene_frame_list.sort()
        imgs_dyna_perscene = []
        for f_idx in range(len(dyna_scene_frame_list)):
            sceneframe_path = dyna_dir_path + '/' + dyna_scene_frame_list[f_idx]
            if 'mat' not in sceneframe_path:
                continue
            try:
                img_dict = sio.loadmat(sceneframe_path)
                need_transpose = False
            except NotImplementedError:
                img_dict = h5py.File(sceneframe_path, 'r')
                need_transpose = True
            except Exception as e:
                print(f"Error loading {sceneframe_path}: {e}")
                raise e
            if "img" in img_dict:
                img = img_dict['img'][:] / 65536.
                if need_transpose:
                    img = img.transpose(2, 1, 0)
                img = np.clip(img, 0, 1)
                img = img.astype(np.float32)
                imgs_dyna_perscene.append(img)
                print('Test Dynamic scene {}: {}/{} is loaded.'.format(i, dyna_scene_list[i], dyna_scene_frame_list[f_idx]))
        imgs_dyna.append(imgs_dyna_perscene)

    if path_test_static is not None:
        test_data_class = {
            'imgs_static': imgs_static,
            'imgs_dyna': imgs_dyna
        }
    else:
        test_data_class = {
            'imgs_dyna': imgs_dyna
        }

    return test_data_class


def resize_images_bilateral(img, target_shape):
    """
    Resize each spectral band using bilinear interpolation.

    Args:
        img (numpy.ndarray): Input image array of shape (H, W, D).
        target_shape (tuple): Target shape as (new_H, new_W).

    Returns:
        numpy.ndarray: Resized image array of shape (new_H, new_W, D).
    """
    new_height, new_width = target_shape
    original_height, original_width, depth = img.shape

    resized_img = np.zeros((new_height, new_width, depth))

    for i in range(depth):
        image = img[:, :, i]
        resized_img[:, :, i] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_img


# ============================================================================
# Test Data Preparation
# ============================================================================

def expand_list_tail(list_ori, n, m):
    """Expand the last n elements of a list by repeating each m times.

    Used to replicate test crop parameters for dynamic scenes that are
    evaluated over multiple temporal windows.

    Args:
        list_ori (list): Original list.
        n (int): Number of trailing elements to expand.
        m (int): Repetition factor for each trailing element.

    Returns:
        list: New list with trailing elements repeated.
    """
    last_n_elements = list_ori[-n:]
    modified_list = list_ori[:-n] + [x for element in last_n_elements for x in [element] * m]
    return modified_list


def prepare_test_gt(test_gt_class, opt, test_crop_params, crop_size=256, mode='train', dyna_sequence=10):
    """Prepare ground-truth test data by cropping scenes to fixed positions.

    For mode='train', each dynamic scene produces 1 clip starting at frame 3.
    For mode='test', each dynamic scene produces multiple non-overlapping clips.

    Args:
        test_gt_class (dict): Test dataset from LoadTest_dyna().
        opt: Configuration namespace.
        test_crop_params (dict): Crop positions for each test scene.
        crop_size (int): Spatial crop size (default 256).
        mode (str): 'train' or 'test'.
        dyna_sequence (int): Total frames per dynamic scene (default 10).

    Returns:
        torch.Tensor: Shape (N_scenes, F, C, H, W), float32.
    """
    nC = opt.SpectralBands
    imgs_static = test_gt_class['imgs_static']
    imgs_dyna = test_gt_class['imgs_dyna']
    test_h_step = test_crop_params['test_h_step']
    test_w_step = test_crop_params['test_w_step']
    test_flip = test_crop_params['test_flip']
    test_h_starts = test_crop_params['test_h_starts']
    test_w_starts = test_crop_params['test_w_starts']

    num_static = len(imgs_static)
    num_dyna = len(imgs_dyna)

    lengths_params = [num_static+num_dyna, len(test_h_step), len(test_w_step), len(test_flip), len(test_h_starts), len(test_w_starts)]

    if len(set(lengths_params)) != 1:
        raise ValueError("Input error: inconsistent lengths!")

    if mode == 'test':
        dyna_sequence = 10
        frame_num_test = opt.frame_num_test
        frame_begin_idx_list = list(range(0, dyna_sequence - frame_num_test, frame_num_test))
        process_times = len(frame_begin_idx_list)
        test_h_step = expand_list_tail(test_h_step, num_dyna, process_times)
        test_w_step = expand_list_tail(test_w_step, num_dyna, process_times)
        test_flip = expand_list_tail(test_flip, num_dyna, process_times)
        test_h_starts = expand_list_tail(test_h_starts, num_dyna, process_times)
        test_w_starts = expand_list_tail(test_w_starts, num_dyna, process_times)
    else:
        process_times = 1
        frame_begin_idx_list = [3]


    processed_data = np.zeros((len(test_h_step), opt.frame_num_test, crop_size, crop_size, nC), dtype=np.float32)
    count_params = 0
    for i in range(num_static):
        data = imgs_static[i]
        processed_data[count_params, ...] = crop_test_dependon_params(data,
                                                                      opt,
                                                                      crop_size,
                                                                      test_h_step[count_params],
                                                                      test_w_step[count_params],
                                                                      test_flip[count_params],
                                                                      test_h_starts[count_params],
                                                                      test_w_starts[count_params])
        count_params = count_params + 1

    for i in range(num_dyna):
        data = imgs_dyna[i]
        for process_idx in range(process_times):
            processed_data[count_params, ...] = crop_test_dependon_params(data,
                                                                          opt,
                                                                          crop_size,
                                                                          test_h_step[count_params],
                                                                          test_w_step[count_params],
                                                                          test_flip[count_params],
                                                                          test_h_starts[count_params],
                                                                          test_w_starts[count_params],
                                                                          dyna_start_frame=frame_begin_idx_list[process_idx])
            count_params = count_params + 1
    test_data = torch.from_numpy(np.transpose(processed_data, (0, 1, 4, 2, 3)))
    return test_data


def crop_test_dependon_params(data, opt, crop_size, test_h_step, test_w_step, test_flip, test_h_starts, test_w_starts, dyna_start_frame=3):
    """Crop a single test scene using predetermined crop parameters.

    Args:
        data: Either a single ndarray(H,W,C) for static scenes, or a list of
              ndarray(H,W,C) for dynamic scenes (one per frame).
        opt: Configuration namespace (uses opt.frame_num_test).
        crop_size (int): Target spatial crop size.
        test_h_step, test_w_step: Per-frame motion offsets (ndarray).
        test_flip (tuple): (flip_h, flip_w) boolean flags.
        test_h_starts, test_w_starts (int): Top-left corner of the crop.
        dyna_start_frame (int): Starting frame index for dynamic scenes.

    Returns:
        ndarray: Shape (F, crop_size, crop_size, C).
    """
    frame_num = opt.frame_num_test
    if not isinstance(data, list):
        img_video = np.tile(data, (frame_num, 1, 1, 1))
    else:
        img_video = np.stack(data[dyna_start_frame: dyna_start_frame + frame_num], axis=0)

    set_crop_param = (test_h_step, test_w_step, test_flip, test_h_starts, test_w_starts)
    process_data = video_crop(img_video,  crop_size, opt, scene_state='imgs_dyna', set_crop_param=set_crop_param)

    return process_data # f, h, w, c


# ============================================================================
# Training Data Augmentation
# ============================================================================

def video_crop(img_video, crop_size, opt, scene_state='imgs_dyna', set_crop_param=None, notmove=False):
    """Randomly crop a video sequence with optional inter-frame motion.

    Simulates camera or object motion by applying cumulative random pixel
    shifts between consecutive frames before cropping.

    Args:
        img_video (ndarray): Shape (F, H, W, C) — input video frames.
        crop_size (int): Target spatial crop size.
        opt: Configuration namespace (uses move_step, dyna_not_move_probability).
        scene_state (str): 'imgs_static' or 'imgs_dyna'.
        set_crop_param (tuple or None): If provided, override random crop with
            (h_steps, w_steps, flip_flags, h_start, w_start).
        notmove (bool): If True, force zero motion (for static scenes).

    Returns:
        ndarray: Cropped video of shape (F, crop_size, crop_size, C).
    """
    f, h, w, c = img_video.shape
    random_step_h = np.random.randint(opt.move_step[0], opt.move_step[1]+1, size=f-1)
    random_step_w = np.random.randint(opt.move_step[0], opt.move_step[1]+1, size=f-1)

    if random.uniform(0, 1) <= 0.5:
        flip_h_flag = True
    else:
        flip_h_flag = False

    if random.uniform(0, 1) <= 0.5:
        flip_w_flag = True
    else:
        flip_w_flag = False

    if scene_state == 'imgs_dyna':
        if random.uniform(0, 1) <= opt.dyna_not_move_probability:
            random_step_h = np.zeros(f-1, dtype=int)
            random_step_w = np.zeros(f-1, dtype=int)
            flip_h_flag = False
            flip_w_flag = False

    if notmove:
        random_step_h = np.zeros(f - 1, dtype=int)
        random_step_w = np.zeros(f - 1, dtype=int)
        flip_h_flag = False
        flip_w_flag = False

    random_step_h_sum = np.cumsum(random_step_h)
    random_step_w_sum = np.cumsum(random_step_w)
    random_step_h_sum = np.concatenate(([0], random_step_h_sum)).astype(int)
    random_step_w_sum = np.concatenate(([0], random_step_w_sum)).astype(int)

    h_index = np.random.randint(0, h - crop_size - max(random_step_h_sum))
    w_index = np.random.randint(0, w - crop_size - max(random_step_w_sum))

    if set_crop_param is not None:
        (random_step_h_sum, random_step_w_sum, flip_flag_set, h_index, w_index) = set_crop_param
        flip_h_flag = flip_flag_set[0]
        flip_w_flag = flip_flag_set[1]

    # Apply flip and crop each frame with its motion offset
    if flip_h_flag:
        img_video = np.flip(img_video, axis=1)
    if flip_w_flag:
        img_video = np.flip(img_video, axis=2)

    per_scene_data = np.zeros((f, crop_size, crop_size, c), dtype=np.float32)

    for f_idx in range(f):
        per_scene_data[f_idx, ...] = img_video[f_idx,
                                     h_index+random_step_h_sum[f_idx]: h_index+random_step_h_sum[f_idx]+crop_size,
                                     w_index+random_step_w_sum[f_idx]: w_index+random_step_w_sum[f_idx]+crop_size, :]
    if flip_h_flag:
        per_scene_data = np.flip(per_scene_data, axis=1)
    if flip_w_flag:
        per_scene_data = np.flip(per_scene_data, axis=2)

    return per_scene_data


def shuffle_crop_dyna_random(train_data, batch_size, frame_num, opt, argument=True):
    """Sample a batch of training video clips with random augmentation.

    Randomly selects static or dynamic scenes (controlled by opt.static_probability),
    crops spatial patches with simulated inter-frame motion, and applies random
    rotation/flip augmentation.

    Args:
        train_data (dict): Training dataset from LoadTraining_dyna().
        batch_size (int): Number of clips to sample.
        frame_num (int): Number of temporal frames per clip.
        opt: Configuration namespace.
        argument (bool): Whether to apply data augmentation (default True).

    Returns:
        torch.Tensor: Shape (B, F, C, crop_size, crop_size) on CUDA.
    """
    crop_size = opt.crop_size
    nC = opt.SpectralBands
    if argument:
        gt_batch = []

        for i in range(batch_size):
            if random.uniform(0, 1) <= 1:
                processed_temp = np.zeros((frame_num, crop_size, crop_size, nC), dtype=np.float32)
                if random.uniform(0, 1) <= opt.static_probability:
                    scene_state = 'imgs_static'
                    index = np.random.choice(range(len(train_data[scene_state])), 1)
                    img_temp = train_data[scene_state][index[0]]
                    img_video = np.tile(img_temp, (frame_num, 1, 1, 1))
                else:
                    scene_state = 'imgs_dyna'
                    index = np.random.choice(range(len(train_data[scene_state])), 1)
                    img_temp = train_data[scene_state][index[0]]
                    start_frame = np.random.randint(0, len(train_data[scene_state][index[0]]) - frame_num + 1)
                    img_video = np.stack(img_temp[start_frame: start_frame + frame_num], axis=0)

                processed_temp[:, :, :, :] = video_crop(img_video, crop_size, opt, scene_state)

                processed_data = torch.Tensor(np.transpose(processed_temp, (0, 3, 1, 2))).cuda().float()
                gt_batch.append(augment_video(processed_data))
            else:
                processed_data_temp = np.zeros((4, frame_num, 128, 128, nC), dtype=np.float32)
                for j in range(4):
                    if random.uniform(0, 1) <= opt.static_probability:
                        scene_state = 'imgs_static'
                        index = np.random.choice(range(len(train_data[scene_state])), 1)
                        img_temp = train_data[scene_state][index[0]]
                        img_video = np.tile(img_temp, (frame_num, 1, 1, 1))
                    else:
                        scene_state = 'imgs_dyna'
                        index = np.random.choice(range(len(train_data[scene_state])), 1)
                        img_temp = train_data[scene_state][index[0]]
                        start_frame = np.random.randint(0, len(train_data[scene_state][index[0]]) - frame_num + 1)
                        img_video = np.stack(img_temp[start_frame: start_frame + frame_num], axis=0)
                    processed_data_temp[j] = video_crop(img_video, crop_size//2, opt, scene_state)
                gt_batch_2 = torch.Tensor(np.transpose(processed_data_temp, (0, 1, 4, 2, 3))).cuda()  # [4, f,28,128,128]
                gt_batch.append(augment_mosaic_video(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, nC), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size,
                                         :]
        gt_batch = torch.Tensor(np.transpose(processed_data, (0, 3, 1, 2)))
        return gt_batch


def augment_video(x):
    """Apply random rotation and flip augmentation to a video tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (F, C, H, W).

    Returns:
        torch.Tensor: Augmented tensor of same shape (F, C, H, W).
    """
    rot_times = random.randint(0, 3)
    v_flip = random.randint(0, 1)
    h_flip = random.randint(0, 1)
    # Random rotation in the (H, W) plane
    for j in range(rot_times):
        x = torch.rot90(x, 1, [2, 3])
    # Random vertical flip (along H axis, dim=2)
    for j in range(v_flip):
        x = torch.flip(x, [2])
    # Random horizontal flip (along W axis, dim=3)
    for j in range(h_flip):
        x = torch.flip(x, [3])
    return x


def augment_mosaic_video(patches):
    """Assemble 4 quarter-size video patches into one full-size video frame.

    Used as an augmentation strategy: independently sample 4 patches of size
    (F, C, H/2, W/2) and stitch them into a single (F, C, H, W) tensor.

    Args:
        patches (torch.Tensor): Shape (4, F, C, H/2, W/2) — four quarter patches.

    Returns:
        torch.Tensor: Stitched tensor of shape (F, C, H, W).
    """
    _, f, c, h, w = patches.shape
    h = h * 2
    w = w * 2
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(f, c, h, w).cuda()
    output_img[:, :, :divid_point_h, :divid_point_w] = patches[0]
    output_img[:, :, :divid_point_h, divid_point_w:] = patches[1]
    output_img[:, :, divid_point_h:, :divid_point_w] = patches[2]
    output_img[:, :, divid_point_h:, divid_point_w:] = patches[3]
    return output_img


# ============================================================================
# Spectral Shift Operations
# ============================================================================

def shift(inputs, step=1):
    """Apply dispersive spectral shift (forward model encoding).

    Each spectral band is shifted by (band_index * step) pixels along the
    column axis, simulating single-dispersion prism encoding.

    Args:
        inputs (torch.Tensor): Shape (B, nC, H, W).
        step (int): Pixels shifted per band.

    Returns:
        torch.Tensor: Shape (B, nC, H, W + (nC-1)*step).
    """
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).float().to(inputs)
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output


def shift_n(inputs, step=1):
    """Apply sub-pixel dispersive shift with step-wise band expansion.

    Expands nC bands into nC*step sub-bands, each shifted by 1 pixel,
    modeling sub-pixel dispersion for step >= 2.

    Args:
        inputs (torch.Tensor): Shape (B, nC, H, W).
        step (int): Sub-pixel expansion factor.

    Returns:
        torch.Tensor: Shape (B, nC*step, H, W + nC*step - 1).
    """
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, step*nC, row, col + nC * step - 1).float().to(inputs)
    for i in range(step*nC):
        output[:, i, :, i: i + col] = inputs[:, math.floor(i/step), :, :]
    return output


def shift_back(inputs, nC, step=1):
    """Reverse dispersive shift on a 2D measurement to recover 3D spectral cube.

    Inverse of shift(): extracts each band from the shifted measurement.

    Args:
        inputs (torch.Tensor): Shape (B, H, W_shifted) — 2D measurement.
        nC (int): Number of spectral bands.
        step (int): Pixels shifted per band.

    Returns:
        torch.Tensor: Shape (B, nC, H, W) where W = W_shifted - (nC-1)*step.
    """
    [bs, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output


def shift_3d_back(inputs, step=1):
    """Reverse dispersive shift on a 3D masked cube.

    Each band is extracted from its shifted position in a per-band manner.

    Args:
        inputs (torch.Tensor): Shape (B, nC, H, W_shifted).
        step (int): Pixels shifted per band.

    Returns:
        torch.Tensor: Shape (B, nC, H, W) where W = W_shifted - (nC-1)*step.
    """
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).float().to(inputs)
    for i in range(nC):
        output[:, i, :, :] = inputs[:, i, :, step * i:step * i + col - (nC - 1) * step]
    return output


# ============================================================================
# Mask Helper Functions
# ============================================================================

def shiftmask2mask(mask3d_batch, opt, step=-1):
    """Convert a shifted (dispersed) 3D mask back to the virtual (un-shifted) domain.

    Args:
        mask3d_batch (torch.Tensor): Shape (B, nC, H, W_shifted).
        opt: Configuration namespace (uses opt.step).
        step (int): Override dispersion step; -1 uses opt.step.

    Returns:
        torch.Tensor: Virtual mask of shape (B, nC, H, H).
    """
    batch_size_frame, nC, mask_size, _ = mask3d_batch.shape
    if step < 0:
        step = opt.step
    mask3d_batch_virtual = torch.zeros(batch_size_frame, nC, mask_size, mask_size).to(mask3d_batch)
    for i in range(nC):
        mask3d_batch_virtual[:, i, :, :] = mask3d_batch[:, i, :, step * i: step * i + mask_size]
    return mask3d_batch_virtual


def mask2shiftmask(mask3d_batch_virtual, opt):
    """Convert a virtual (un-shifted) mask to the shifted (dispersed) domain.

    Inverse of shiftmask2mask().

    Args:
        mask3d_batch_virtual (torch.Tensor): Shape (B, nC, H, H).
        opt: Configuration namespace (uses opt.step).

    Returns:
        torch.Tensor: Shifted mask of shape (B, nC, H, H + (nC-1)*step).
    """
    batch_size_frame, nC, mask_size, _ = mask3d_batch_virtual.shape
    step = opt.step
    mask3d_batch = torch.zeros(batch_size_frame, nC, mask_size, mask_size+(nC-1)*step).to(mask3d_batch_virtual)
    for i in range(nC):
        mask3d_batch[:, i, :, step * i: step * i + mask_size] = mask3d_batch_virtual[:, i, :, :]
    return mask3d_batch


# ============================================================================
# Mask Initialization Functions
# ============================================================================

def init_sdcassi_mask(batch_size, opt):
    """Initialize SD-CASSI random binary mask from file.

    Loads a 2D binary mask and replicates it across spectral bands.
    For SD-CASSI, the mask is cropped with an offset to center the
    dispersion range within the detector.

    Args:
        batch_size (int): Batch dimension for the mask tensor.
        opt: Configuration namespace.

    Returns:
        tuple: (mask3d_batch, input_mask)
            - mask3d_batch: Shape (B, nC, H, W) — 3D coded aperture.
            - input_mask: Network-ready mask representation (depends on opt.input_mask).
    """
    mask_size = opt.crop_size
    step = opt.step
    nC = opt.SpectralBands
    if not os.path.exists(opt.mask_path + '/mask_dd_2d.mat'):
        raise FileNotFoundError(f"Mask file not found at {opt.mask_path}")
    mask_2d = sio.loadmat(opt.mask_path + '/mask_dd_2d.mat')['mask_dd_2d']
    mask_2d = mask_2d[:, math.floor(nC/2)*step: mask_size + math.floor(nC/2)*step]
    mask_2d = torch.from_numpy(mask_2d)
    [H, W] = mask_2d.shape
    mask3d_batch = mask_2d.expand([batch_size, nC, H, W]).float()

    if opt.input_mask == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch, step=opt.step)
        input_mask = shift_mask3d_batch
    elif opt.input_mask == 'Phi_PhiPhiT':
        Phi_batch = shift(mask3d_batch, step=opt.step)
        Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
        Phi_s_batch[Phi_s_batch == 0] = 1
        input_mask = (Phi_batch, Phi_s_batch)
    elif opt.input_mask == 'Mask':
        input_mask = mask3d_batch
    elif opt.input_mask == 'None':
        input_mask = None
    return mask3d_batch, input_mask


def init_notch_mask_dd(batch_size, opt):
    """Initialize DD-CASSI notch filter mask (synthetically generated).

    Creates a periodic notch pattern where most pixels transmit and narrow
    notch lines block light, with row-dependent cyclic shifts.

    Args:
        batch_size (int): Batch dimension.
        opt: Configuration namespace.

    Returns:
        tuple: (mask3d_batch, input_mask) — same format as init_sdcassi_mask.
    """
    mask_size = 256
    nC = opt.SpectralBands
    step = opt.step

    mask_batch = torch.ones(batch_size, mask_size, mask_size + step * (nC - 1))

    mask_gap_x = 4
    mask_gap_y = nC * step

    mask_batch[:, :, 1:mask_size + step * (nC - 1) :mask_gap_y] = 0
    for i in range(0, mask_size):
        mask_batch[:, i, :] = torch.roll(mask_batch[:, i, :], (i % mask_gap_x) * (mask_gap_y // mask_gap_x))
        mask_batch[:, i, :(i % mask_gap_x) * (mask_gap_y // mask_gap_x)] = 1

    mask3d_batch = mask_batch.unsqueeze(1).repeat(1, nC, 1, 1)

    mask3d_batch_virtual = torch.zeros(batch_size, nC, mask_size, mask_size)
    for i in range(nC):
        mask3d_batch_virtual[:, i, :, :] = mask3d_batch[:, i, :, step * i: step * i + mask_size]

    if opt.input_mask == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch_virtual, step=opt.step)
        input_mask = shift_mask3d_batch
    elif opt.input_mask == 'Phi_PhiPhiT':
        Phi_batch = mask3d_batch_virtual
        Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
        Phi_s_batch[Phi_s_batch == 0] = 1
        input_mask = (Phi_batch, Phi_s_batch)
    elif opt.input_mask == 'Mask':
        input_mask = mask3d_batch_virtual
    elif opt.input_mask is None:
        input_mask = None

    return mask3d_batch, input_mask


def init_pmvis_mask_sd(batch_size, opt):
    """Initialize SD-CASSI PMVIS (sparse uniform) pattern mask.

    Args:
        batch_size (int): Batch dimension.
        opt: Configuration namespace.

    Returns:
        tuple: (mask3d_batch, input_mask) — same format as init_sdcassi_mask.
    """
    mask_size = 256
    nC = opt.SpectralBands
    step = opt.step
    mask_batch = torch.zeros(batch_size, mask_size, mask_size).float()

    mask_gap_x = 4
    mask_gap_y = nC * step

    mask_batch[:, :, 1:mask_size:mask_gap_y] = 1
    for i in range(0, mask_size):
        mask_batch[:, i, :] = torch.roll(mask_batch[:, i, :], (i % mask_gap_x) * (mask_gap_y // mask_gap_x))
        mask_batch[:, i, :(i % mask_gap_x) * (mask_gap_y // mask_gap_x)] = 0

    mask3d_batch = mask_batch.unsqueeze(1).repeat(1, nC, 1, 1)

    if opt.input_mask == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch, step=opt.step)
        input_mask = shift_mask3d_batch
    elif opt.input_mask == 'Phi_PhiPhiT':
        Phi_batch = shift(mask3d_batch, step=opt.step)
        Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
        Phi_s_batch[Phi_s_batch == 0] = 1
        input_mask = (Phi_batch, Phi_s_batch)

    return mask3d_batch, input_mask


def init_ddcassi_mask(batch_size, opt):
    """Initialize DD-CASSI random binary mask from file.

    For DD-CASSI, the mask is loaded at full width (mask_size + (nC-1)*step)
    since dual dispersion requires the extended spatial footprint.

    Args:
        batch_size (int): Batch dimension.
        opt: Configuration namespace.

    Returns:
        tuple: (mask3d_batch, input_mask) — same format as init_sdcassi_mask.
    """
    mask_size = opt.mask_size
    step = opt.step
    nC = opt.SpectralBands

    if not os.path.exists(opt.mask_path + '/mask_dd_binary.mat'):
        raise FileNotFoundError(f"Mask file not found at {opt.mask_path}")
    mask_2d = sio.loadmat(opt.mask_path + '/mask_dd_binary.mat')['binary_mask']
    mask_2d = mask_2d[:, :mask_size + (nC - 1) * step]
    mask_2d = torch.from_numpy(mask_2d)
    [H, W] = mask_2d.shape
    mask3d_batch = mask_2d.expand([batch_size, nC, H, W]).float()

    mask3d_batch_virtual = torch.zeros(batch_size, nC, mask_size, mask_size)
    for i in range(nC):
        mask3d_batch_virtual[:, i, :, :] = mask3d_batch[:, i, :, step * i: step * i + mask_size]

    if opt.input_mask == 'Phi':
        input_mask = mask3d_batch_virtual   # dd no shift
    elif opt.input_mask == 'Phi_PhiPhiT':
        Phi_batch = mask3d_batch_virtual
        Phi_s_batch = torch.sum(Phi_batch ** 2, 1)
        Phi_s_batch[Phi_s_batch == 0] = 1
        input_mask = (Phi_batch, Phi_s_batch)
    elif opt.input_mask == 'Mask':
        input_mask = mask3d_batch_virtual
    elif opt.input_mask == 'None':
        input_mask = None

    return mask3d_batch, input_mask


# ============================================================================
# Unified Mask Dispatcher
# ============================================================================

def init_mask(batch_size, opt):
    """Initialize coded aperture mask based on imaging system configuration.

    Dispatches to the appropriate mask initialization function based on
    opt.dispersion ('single'/'dual') and opt.mask_pattern ('random'/'notch'/'pmvis').

    Args:
        batch_size (int): Batch dimension for the mask tensor.
        opt: Configuration namespace.

    Returns:
        tuple: (mask3d_batch, input_mask)
            - mask3d_batch: Shape (B, nC, H, W_mask).
            - input_mask: Network-ready representation, typically (Phi, Phi_s) tuple.

    Raises:
        ValueError: If the dispersion/mask_pattern combination is unsupported.
    """
    if opt.dispersion == 'single':
        if opt.mask_pattern == 'random':
            return init_sdcassi_mask(batch_size, opt)
        elif opt.mask_pattern == 'pmvis':
            return init_pmvis_mask_sd(batch_size, opt)
    elif opt.dispersion == 'dual':
        if opt.mask_pattern == 'random':
            return init_ddcassi_mask(batch_size, opt)
        elif opt.mask_pattern == 'notch':
            return init_notch_mask_dd(batch_size, opt)
    raise ValueError(f"Unsupported combination: dispersion={opt.dispersion}, mask_pattern={opt.mask_pattern}")


def init_mask_video(batch_size, opt, mode='train'):
    """Initialize masks replicated across temporal frames.

    Calls init_mask() then repeats the mask along the frame dimension,
    reshaping to (B*F, ...) for batched forward model computation.

    Args:
        batch_size (int): Number of scenes (before frame expansion).
        opt: Configuration namespace.
        mode (str): 'train' uses opt.frame_num, 'test' uses opt.frame_num_test.

    Returns:
        tuple: (mask3d_batch, input_mask) — same as init_mask but with
               batch dimension expanded to batch_size * frame_num.
    """
    if mode == 'train':
        frame_num = opt.frame_num
    elif mode == 'test':
        frame_num = opt.frame_num_test

    mask3d_batch, input_mask = init_mask(batch_size, opt)

    b, c, h, w = mask3d_batch.shape
    mask3d_batch = mask3d_batch.unsqueeze(dim=1).repeat(1, frame_num, 1, 1, 1).reshape(batch_size*frame_num, c, h, w)

    if input_mask is not None:
        if isinstance(input_mask, tuple):
            (Phi_batch, Phi_s_batch) = input_mask
            b, c, h, w = Phi_batch.shape
            Phi_batch = Phi_batch.unsqueeze(dim=1).repeat(1, frame_num, 1, 1, 1).reshape(batch_size * frame_num, c, h, w)
            b, h, w = Phi_s_batch.shape
            Phi_s_batch = Phi_s_batch.unsqueeze(dim=1).repeat(1, frame_num, 1, 1).reshape(batch_size * frame_num, h, w)
            input_mask = (Phi_batch, Phi_s_batch)
        else:
            b, c, h, w = input_mask.shape
            input_mask = input_mask.unsqueeze(dim=1).repeat(1, frame_num, 1, 1, 1).reshape(batch_size*frame_num, c, h, w)
    return mask3d_batch, input_mask


# ============================================================================
# Measurement Generation
# ============================================================================

def gen_meas_singledisp(opt, data_batch, mask3d_batch, Y2H=True, mul_mask=False, mean_meas=False):
    """Generate measurement using the single-dispersion (SD-CASSI) forward model.

    Forward model: Y = sum_lambda( shift_n(Mask * X, step) ) / step
    If Y2H=True, the pseudo-inverse initial estimate H is computed via shift_back.

    Args:
        opt: Configuration namespace.
        data_batch (torch.Tensor): Ground truth spectral cube (B, nC, H, W).
        mask3d_batch (torch.Tensor): 3D coded aperture mask (B, nC, H, W).
        Y2H (bool): If True, return pseudo-inverse estimate H; else return raw measurement.
        mul_mask (bool): If True, multiply H by the mask to get HM.
        mean_meas (bool): If True, normalize measurement by band count.

    Returns:
        torch.Tensor: H (B, nC, H, W), HM (B, nC, H, W), or meas (B, H, W).
    """
    nC = data_batch.shape[1]

    temp = shift_n(mask3d_batch * data_batch, step=opt.step)
    meas = torch.sum(temp, 1, keepdim=False) / opt.step
    meas = meas[..., :-opt.step + 1]

    _, h_ori, w_ori = meas.shape

    if mean_meas:
        meas = meas / nC * 2

    if Y2H:
        meas = F.interpolate(meas.unsqueeze(1), size=(h_ori, w_ori), mode='bicubic').squeeze()
        H = shift_back(meas, nC=nC, step=opt.step)
        H = H / nC * 2
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas


def gen_meas_dualdisp(opt, data_batch, mask3d_batch, Y2H=True, mul_mask=False, mean_meas=False):
    """Generate measurement using the dual-dispersion (DD-CASSI) forward model.

    For step==1: Y = sum( shift_3d_back(Mask * shift(X)) )
    For step>=2: uses multi-step sub-pixel dispersion model with mask propagation.
    If Y2H=True, the pseudo-inverse H is computed by replicating Y across bands.

    Args:
        opt: Configuration namespace.
        data_batch (torch.Tensor): Ground truth spectral cube (B, nC, H, W).
        mask3d_batch (torch.Tensor): 3D coded aperture mask (B, nC, H, W_mask).
        Y2H (bool): If True, return pseudo-inverse estimate H; else return raw measurement.
        mul_mask (bool): If True, multiply H by the mask to get HM.
        mean_meas (bool): If True, normalize measurement by band count.

    Returns:
        torch.Tensor: H (B, nC, H, W), HM (B, nC, H, W), or meas (B, H, W).
    """
    b, nC, crop_size, _ = data_batch.shape
    _, _, masksize_h, masksize_w = mask3d_batch.shape

    if opt.step == 1:
        # NOTE: step==1 branch does not apply mean_meas normalization,
        # consistent with the original implementation. Default config uses step=2.
        temp = (mask3d_batch.float().to(data_batch)) * shift(data_batch, opt.step)
        meas = torch.sum(shift_3d_back(temp, step=opt.step), dim=1, keepdim=False)
    elif opt.step >= 2:
        temp_mask_class = []
        temp_mask = mask3d_batch.float().to(data_batch)
        shifted_tensor = temp_mask.clone()

        mask_forward_one_step = shiftmask2mask(shifted_tensor, opt)
        for i in range(opt.step):
            if i >= 1:
                padding_information = torch.zeros_like(mask_forward_one_step[:, :, :, -(opt.step-1)].unsqueeze(dim=-1))
                padding_information[:, :-1, :, :] = mask_forward_one_step[:, 1:, :, -(opt.step-1)].unsqueeze(dim=-1)
                mask_forward_one_step = torch.cat([mask_forward_one_step[:, :, :, 1:], padding_information], dim=-1)
            shifted_tensor = mask2shiftmask(mask_forward_one_step, opt)
            shifted_tensor_temp = torch.cat((torch.zeros_like(shifted_tensor[:, :, :, :i]),
                                             shifted_tensor,
                                             torch.zeros_like(shifted_tensor[:, :, :, -opt.step+i:-1])), dim=-1)
            temp_mask_class.append(shifted_tensor_temp)

        mask3d_batch_multi_step = torch.stack(temp_mask_class, dim=2)
        mask3d_batch_multi_step = mask3d_batch_multi_step.reshape([b, nC*opt.step]+list(mask3d_batch_multi_step.shape[3:]))

        shift_data_batch = shift_n(data_batch, opt.step)
        temp_shift_class = mask3d_batch_multi_step * shift_data_batch
        meas = torch.sum(shift_3d_back(temp_shift_class, step=1), dim=1, keepdim=False) / opt.step

        if mean_meas:
            meas = meas / nC * 2

    if Y2H:
        meas = meas.unsqueeze(1)
        H = meas.repeat(1, nC, 1, 1)
        if not mean_meas:
            H = H / nC * 2

        if mul_mask:
            HM = torch.zeros_like(H)
            [_, _, _, col] = HM.shape
            for i in range(nC):
                HM[:, i, :, :] = torch.mul(H[:, i, :, :], mask3d_batch[:, i, :, i:i + col].float().to(data_batch))
            return HM
        return H
    return meas


def init_meas(opt, gt, mask, mean_meas):
    """Generate measurement and initial estimate from ground truth.

    Dispatches to single- or dual-dispersion forward model based on
    opt.dispersion, with input representation controlled by opt.input_setting.

    Args:
        opt: Configuration namespace.
        gt (torch.Tensor): Ground truth spectral cube (B*F, nC, H, W).
        mask (torch.Tensor): 3D coded aperture mask (B*F, nC, H, W_mask).
        mean_meas (bool): Whether to normalize measurement.

    Returns:
        torch.Tensor: Initial estimate for the network, shape (B*F, nC, H, W).
    """
    if opt.dispersion == 'dual':
        if opt.input_setting == 'H':
            return gen_meas_dualdisp(opt, gt, mask, Y2H=True, mul_mask=False, mean_meas=mean_meas)
        elif opt.input_setting == 'HM':
            return gen_meas_dualdisp(opt, gt, mask, Y2H=True, mul_mask=True, mean_meas=mean_meas)
        elif opt.input_setting == 'Y':
            return gen_meas_dualdisp(opt, gt, mask, Y2H=False, mul_mask=False, mean_meas=mean_meas)
    elif opt.dispersion == 'single':
        if opt.input_setting == 'H':
            return gen_meas_singledisp(opt, gt, mask, Y2H=True, mul_mask=False, mean_meas=mean_meas)
        elif opt.input_setting == 'HM':
            return gen_meas_singledisp(opt, gt, mask, Y2H=True, mul_mask=True, mean_meas=mean_meas)
        elif opt.input_setting == 'Y':
            return gen_meas_singledisp(opt, gt, mask, Y2H=False, mul_mask=False, mean_meas=mean_meas)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def torch_psnr(img, ref):
    """Compute PSNR averaged over spectral bands (8-bit quantized).

    Args:
        img (torch.Tensor): Reconstructed image (nC, H, W), values in [0, 1].
        ref (torch.Tensor): Reference image (nC, H, W), values in [0, 1].

    Returns:
        torch.Tensor: Scalar PSNR value (dB).
    """
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)
    return psnr / nC


def torch_ssim(img, ref):
    """Compute SSIM between reconstructed and reference images.

    Args:
        img (torch.Tensor): Reconstructed image (nC, H, W).
        ref (torch.Tensor): Reference image (nC, H, W).

    Returns:
        torch.Tensor: Scalar SSIM value.
    """
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def torch_sam(img, ref):
    """Compute Spectral Angle Mapper (SAM) in degrees.

    Reference: https://github.com/MyuLi/SERT

    Args:
        img (torch.Tensor): Reconstructed image (nC, H, W).
        ref (torch.Tensor): Reference image (nC, H, W).

    Returns:
        torch.Tensor: Scalar mean SAM value (degrees).
    """
    sum1 = torch.sum(ref * img, 0)
    sum2 = torch.sum(ref * ref, 0)
    sum3 = torch.sum(img * img, 0)
    t = (sum2 * sum3) ** 0.5
    numlocal = torch.gt(t, 0)
    num = torch.sum(numlocal)
    t = sum1 / t
    angle = torch.acos(t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num
    SAM = averangle * 180 / 3.14159256
    return SAM


# ============================================================================
# Utilities
# ============================================================================

def time2file_name(time):
    """Convert a datetime string to a filesystem-safe filename.

    Args:
        time (str): Datetime string in format 'YYYY-MM-DD HH:MM:SS...'.

    Returns:
        str: Formatted as 'YYYY_MM_DD_HH_MM_SS'.
    """
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def checkpoint(model, epoch, model_path, logger):
    """Save model checkpoint to disk.

    Args:
        model: PyTorch model (may be DataParallel-wrapped).
        epoch (int): Current epoch number.
        model_path (str): Directory to save the checkpoint.
        logger: Logger instance for status output.
    """
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


def dict2str(opt, indent_l=1):
    """Convert a dict (or OrderedDict) to a formatted multi-line string for logging.

    Args:
        opt (dict): Configuration dictionary.
        indent_l (int): Indentation level (default 1).

    Returns:
        str: Formatted string representation.
    """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def write_results_to_txt(psnr_list, ssim_list, sam_list, scene_time_list, strred_list=None,
                         output_file="output.txt", distinguish_static=False,
                         model_property=None, frame_num=3):
    """
    Write PSNR, SSIM, SAM results to a text file.

    Args:
        psnr_list: Per-scene PSNR values (2D list, each scene has multiple frames)
        ssim_list: Per-scene SSIM values (2D list, each scene has multiple frames)
        sam_list: Per-scene SAM values (2D list, each scene has multiple frames)
        scene_time_list: Per-scene inference time
        strred_list: Optional per-scene ST-RRED values
        output_file: Output file path
        distinguish_static: Whether to separately report static vs dynamic scenes
        model_property: Optional (flops_G, n_param) tuple
        frame_num: Number of frames per sequence
    """
    # Get number of scenes
    num_scenes = len(psnr_list)

    # Compute per-scene averages
    psnr_avg_per_scene = [np.mean(psnr_list[i]) for i in range(num_scenes)]
    ssim_avg_per_scene = [np.mean(ssim_list[i]) for i in range(num_scenes)]
    sam_avg_per_scene = [np.mean(sam_list[i]) for i in range(num_scenes)]

    frame_num = len(sam_list[0])

    # Compute overall averages
    psnr_avg_all_scenes = np.mean(psnr_avg_per_scene)
    ssim_avg_all_scenes = np.mean(ssim_avg_per_scene)
    sam_avg_all_scenes = np.mean(sam_avg_per_scene)

    if strred_list is not None:
        strred_avg_all_scenes = np.mean(strred_list)

    if distinguish_static:
        psnr_avg_static_scenes = np.mean(psnr_avg_per_scene[:10])
        ssim_avg_static_scenes = np.mean(ssim_avg_per_scene[:10])
        sam_avg_static_scenes = np.mean(sam_avg_per_scene[:10])
        psnr_avg_dyna_scenes = np.mean(psnr_avg_per_scene[10:])
        ssim_avg_dyna_scenes = np.mean(ssim_avg_per_scene[10:])
        sam_avg_dyna_scenes = np.mean(sam_avg_per_scene[10:])
        if strred_list is not None:
            strred_avg_static_scenes = np.mean(strred_list[:10])
            strred_avg_dyna_scenes = np.mean(strred_list[10:])

    # Write results to file
    with open(output_file, "w") as f:
        # Per-scene per-frame results
        for scene_idx in range(num_scenes):
            f.write(f"scene{scene_idx + 1}:\n")
            for frame_idx in range(frame_num):
                psnr_per_frame = psnr_list[scene_idx][frame_idx]
                ssim_per_frame = ssim_list[scene_idx][frame_idx]
                sam_per_frame = sam_list[scene_idx][frame_idx]
                f.write(f"    frame{frame_idx + 1}: {psnr_per_frame:.2f} {ssim_per_frame:.4f} {sam_per_frame:.4f}\n")

            if strred_list is not None:
                strred_per_seq = strred_list[scene_idx]
                f.write(f"    strred_seq {frame_idx + 1}: {strred_per_seq:.4f}\n")


        # Per-scene averages
        f.write("\n\nave_per_scene:\n")
        for scene_idx in range(num_scenes):
            if strred_list is not None:
                f.write(
                    f"   ave scene{scene_idx + 1}: psnr {psnr_avg_per_scene[scene_idx]:.2f} ssim {ssim_avg_per_scene[scene_idx]:.4f}"
                    f" sam {sam_avg_per_scene[scene_idx]:.4f} strred {strred_list[scene_idx]:.4f}  time: {scene_time_list[scene_idx]:.2f} ms\n")
            else:
                f.write(
                    f"   ave scene{scene_idx + 1}: psnr {psnr_avg_per_scene[scene_idx]:.2f} ssim {ssim_avg_per_scene[scene_idx]:.4f}"
                    f" sam {sam_avg_per_scene[scene_idx]:.4f}  time: {scene_time_list[scene_idx]:.2f} ms\n")

        if distinguish_static:
            f.write("\n\nave_static_scene:\n")
            f.write(f"   psnr: {psnr_avg_static_scenes:.4f}\n")
            f.write(f"   ssim: {ssim_avg_static_scenes:.4f}\n")
            f.write(f"   sam:  {sam_avg_static_scenes:.4f}\n")
            if strred_list is not None:
                f.write(f"   strred:  {strred_avg_static_scenes:.4f}\n")

            f.write("\n\nave_dyna_scene:\n")
            f.write(f"   psnr: {psnr_avg_dyna_scenes:.4f}\n")
            f.write(f"   ssim: {ssim_avg_dyna_scenes:.4f}\n")
            f.write(f"   sam:  {sam_avg_dyna_scenes:.4f}\n")
            if strred_list is not None:
                f.write(f"   strred:  {strred_avg_dyna_scenes:.4f}\n")

            psnr_avg_class = np.mean([psnr_avg_static_scenes, psnr_avg_dyna_scenes])
            ssim_avg_class = np.mean([ssim_avg_static_scenes, ssim_avg_dyna_scenes])
            sam_avg_class = np.mean([sam_avg_static_scenes, sam_avg_dyna_scenes])

            f.write("\n\nave_two_class:\n")
            print("\n\nave_two_class:\n")
            f.write(f"   psnr: {psnr_avg_class:.4f}\n")
            print(f"   psnr: {psnr_avg_class:.4f}\n")
            f.write(f"   ssim: {ssim_avg_class:.4f}\n")
            print(f"   ssim: {ssim_avg_class:.4f}\n")
            f.write(f"   sam:  {sam_avg_class:.4f}\n")
            print(f"   sam:  {sam_avg_class:.4f}\n")
            if strred_list is not None:
                strred_avg_class = np.mean([strred_avg_static_scenes, strred_avg_dyna_scenes])
                f.write(f"   strred:  {strred_avg_class:.4f}\n")
                print(f"   strred:  {strred_avg_class:.4f}\n")

        # Overall averages
        f.write("\n\nave_all_scene:\n")
        f.write(f"   psnr: {psnr_avg_all_scenes:.4f}\n")
        f.write(f"   ssim: {ssim_avg_all_scenes:.4f}\n")
        f.write(f"   sam:  {sam_avg_all_scenes:.4f}\n")
        if strred_list is not None:
            f.write(f"   strred:  {strred_avg_all_scenes:.4f}\n")
        f.write(f"   time:  {np.mean(scene_time_list):.2f} ms\n")

        if model_property is not None:
            (flops_G, n_param) = model_property

            f.write("\n\nmodel_property:\n")
            f.write(f"   GFLOPs:  {flops_G:.2f} \n")
            f.write(f"   Ave GFLOPs:  {flops_G/frame_num:.2f} \n")
            f.write(f"   Params:  {n_param/1e6:.2f} M\n")


def my_summary_test(test_model, H=256, W=256, C=28, N=1, input=None, mask=None):
    """Compute FLOPs and parameter count for a model.

    Args:
        test_model: The model to profile.
        H, W, C, N: Default spatial/spectral/batch dimensions (unused when input provided).
        input: Actual input tensor for FLOPs analysis.
        mask: Actual mask tensor for FLOPs analysis.

    Returns:
        tuple: (flops_G, n_param) — GFLOPs and total parameter count.
    """
    model = test_model.cuda()
    print(model)
    flops = FlopCountAnalysis(model, (input, mask))
    n_param = sum([p.nelement() for p in model.parameters()])
    flops_G = flops.total() / (1024 * 1024 * 1024)
    print(f'GFLOPs: {flops_G:.2f}')
    print(f'Params: {n_param}')
    return flops_G, n_param
