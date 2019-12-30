import os
import six
import numpy as np
import torch
import lmdb
import pyarrow as pa
from torchvision.transforms import ToTensor
from PIL import Image


def seg_patches_to_tensor(patches):
    """
    Converts an [T, V, W, H, C] temporal patch collection to tensor
    :param patches:
    :return:
    """
    t, v, w, h, c = patches.shape
    patches = patches.reshape(-1, w, h, c)
    patches_tensor = torch.stack([ToTensor()(p) for p in patches])
    patches_tensor = patches_tensor.view(t, v, c, w, h)
    patches_tensor = patches_tensor.permute(2, 0, 1, 3, 4).contiguous()
    return patches_tensor


def get_single_img_patches(img_path, sing_pose_data, lmdb_env=None, patch_size=None):
    """
    Loads a single image's patches. When an lmdb_env is provided, loads from there
    :param img_path:
    :param sing_pose_data:
    :param lmdb_env:
    :param patch_size:
    :return:
    """
    if patch_size is None:
        patch_size = np.array([32, 32])
    elif isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    patch_size = np.array(patch_size, dtype=np.int32) // 2

    if lmdb_env is None:
        img = Image.open(img_path)
    else:
        key = '_'.join(img_path.split('.')[0].split('/')[-2:])  # '/videos/01_003/0692.jpg' -> '01_003_0692'
        key = key.encode('ascii')
        img = img_from_db(lmdb_env, key)

    int_coords = sing_pose_data[:2].transpose().astype(np.int32)
    patch_coords = np.array([int_coords - patch_size, int_coords + patch_size])
    patch_coords = patch_coords.transpose([1, 0, 2]).reshape(-1, 4)
    patches = [img.crop(pc) for pc in patch_coords]
    patches_np = np.stack([np.asarray(i) for i in patches])
    return patches_np


def gen_clip_seg_data_np(clip_dict, start_ofst=0, seg_stride=4, seg_len=12, scene_id='', clip_id='', ret_keys=False):
    """
    Generate an array of segmented sequences, each object is a segment and a corresponding metadata array
    """
    pose_segs_data = []
    pose_segs_meta = []
    person_keys = {}
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        sing_pose_np, sing_pose_meta, sing_pose_keys = single_pose_dict2np(clip_dict, idx)
        key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))
        person_keys[key] = sing_pose_keys

        curr_pose_segs_np, curr_pose_segs_meta = split_pose_to_segments(sing_pose_np, sing_pose_meta, sing_pose_keys,
                                                                        start_ofst, seg_stride, seg_len,
                                                                        scene_id=scene_id, clip_id=clip_id)
        pose_segs_data.append(curr_pose_segs_np)
        pose_segs_meta += curr_pose_segs_meta
    pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)

    del pose_segs_data
    if ret_keys:
        return pose_segs_data_np, pose_segs_meta, person_keys
    else:
        return pose_segs_data_np, pose_segs_meta


def single_pose_dict2np(person_dict, idx):
    single_person = person_dict[str(idx)]
    sing_pose_np = []
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    single_person_dict_keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [index, first_frame]
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
        sing_pose_np.append(curr_pose_np)
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    return sing_pose_np, sing_pose_meta, single_person_dict_keys


def is_single_person_dict_continuous(sing_person_dict):
    """
    Checks if an input clip is continuous or if there are frames missing
    :return:
    """
    start_key = min(sing_person_dict.keys())
    person_dict_items = len(sing_person_dict.keys())
    sorted_seg_keys = sorted(sing_person_dict.keys(), key=lambda x: int(x))
    return is_seg_continuous(sorted_seg_keys, start_key, person_dict_items)


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Checks if an input clip is continuous or if there are frames missing
    :param sorted_seg_keys:
    :param start_key:
    :param seg_len:
    :param missing_th: The number of frames that are allowed to be missing on a sequence,
    i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
    :return:
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def split_pose_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12,
                           scene_id='', clip_id=''):
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_segs_meta = []
    num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(np.int)
    single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])
    return pose_segs_np, pose_segs_meta


def get_seg_patches(img_dir, seg_pose_data, seg_meta, lmdb_env=None, patch_size=None, pre_proc_seg=None):
    """
    Collates a segments patches. Allows reuse of a previous segment with only appending missing frames
    """
    img_list = [img for img in os.listdir(img_dir) if img.endswith('.jpg')]
    fn_prefix = img_list[4].split('.')[0]
    if len(fn_prefix) == 3:
        fmt_str = '{:03d}.jpg'
    else:
        fmt_str = '{:04d}.jpg'

    seg_patches = []
    first_vid_frame = seg_meta[-1]
    if pre_proc_seg is not None:
        first_load_frame = pre_proc_seg.shape[0]
    else:
        first_load_frame = 0
    num_frames = seg_pose_data.shape[1]

    for t in range(first_load_frame, num_frames):
        img_fn = fmt_str.format(first_vid_frame + t)
        img_path = os.path.join(img_dir, img_fn)
        sing_pose_data = seg_pose_data[:2, t]
        curr_frame_patches = get_single_img_patches(img_path, sing_pose_data, lmdb_env=lmdb_env,
                                                    patch_size=patch_size)
        seg_patches.append(curr_frame_patches)
    new_seg_patches = np.stack(seg_patches)
    if pre_proc_seg is not None:
        seg_patches = np.concatenate([pre_proc_seg, new_seg_patches], axis=0)
    else:
        seg_patches = new_seg_patches

    seg_patches = np.transpose(seg_patches, (4, 0, 1, 2, 3))
    seg_patches = seg_patches.astype(np.float32)
    seg_patches /= 255.0
    return seg_patches


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def img_from_db(env, key):
    with env.begin(write=False) as txn:
        unpacked = txn.get(key)
    # load image
    imgbuf = loads_pyarrow(unpacked)  # [0] used when serializing metadata with image
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


def patches_from_db(env, key):
    with env.begin(write=False) as txn:
        patch_buf = txn.get(key)
    # load pose and metadata
    patches, metadata = loads_pyarrow(patch_buf)
    return patches, metadata

