"""
Generate an extracted patch dataset to save time during training. Avoids the need to load and
extract patches on-the-fly which bottlenecks the training process.
"""
import os
import os.path as osp
import time
from PIL import Image

import json
import lmdb
import numpy as np
from utils.data_utils import normalize_pose
from utils.patch_utils import loads_pyarrow, dumps_pyarrow, single_pose_dict2np
from utils.pose_seg_dataset import keypoints17_to_coco18, get_seg_patches


def gen_sing_dataset(person_json_root, num_clips=None, normalize_pose_segs=False,
                     kp18_format=True, vid_res=[856, 480]):
    data_arr = []
    meta_arr = []

    dir_list = os.listdir(person_json_root)
    json_list = sorted([fn for fn in dir_list if fn.endswith('.json')])
    if num_clips:
        json_list = json_list[:num_clips]  # For debugging purposes
    for person_dict_fn in json_list:
        scene_id, clip_id = person_dict_fn.split('_')[:2]
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data, clip_segs_meta = gen_clip_sing_data_np(clip_dict, scene_id=scene_id, clip_id=clip_id)
        data_arr.append(clip_segs_data)
        meta_arr += [clip_segs_meta]
    for i, clip in enumerate(data_arr):
        for j, person in enumerate(clip):
            if normalize_pose_segs:
                data_arr[i][j] = normalize_pose(data_arr[i][j], vid_res=vid_res)
            if kp18_format and person.shape[-2] == 17:
                data_arr[i][j] = keypoints17_to_coco18(data_arr[i][j])
            data_arr[i][j] = np.transpose(data_arr[i][j], (2, 0, 1)).astype(np.float32)
    return data_arr, meta_arr


def extract_patches_2lmdb(pose_dir, vid_dir, split='training', patch_size=(12,12), write_frequency=100):
    """
    First - Generate pose dataset
    Next - Extract patches (An np.array for each figure in each clip)
    Finally - Serialize and save into an LMDB where keys are clip+pose: '01_014_4' - Scene 1, clip 14, person 4
              (Additionally serialize an array of keys and a length value)
    :return:
    """
    # Load clip pose dictionaries
    patches_data, pose_meta = gen_sing_dataset(pose_dir)
    print("Extracted {} clips".format(len(patches_data)))
    size_str = '{}x{}'.format(*patch_size)
    lmdb_path = osp.join(vid_dir, "{}{}.lmdb".format(split, size_str))
    isdir = osp.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    keys = []
    # Extract patches
    idx = 1
    for i, clip in enumerate(patches_data):
        print("Clip {}".format(i))
        for j, person_pose_data in enumerate(clip):
            person_meta = pose_meta[i][j]
            dirname = '{:02d}_{:03d}'.format(*person_meta[:2])
            img_dir = os.path.join(vid_dir, dirname)
            person_patches = get_seg_patches(img_dir, person_pose_data, person_meta, patch_size=patch_size)
            key = ('{:02d}_{:04d}_{:02d}'.format(*person_meta[:3])).encode('ascii')
            keys.append(key)
            txn.put(key, dumps_pyarrow([person_patches, person_meta]))
            if idx % write_frequency == 0:
                print("[%d/%d]" % (i, len(patches_data)))
                txn.commit()
                txn = db.begin(write=True)
                idx = 0
            idx += 1

    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
    print("Done")


def gen_clip_sing_data_np(clip_dict, scene_id=None, clip_id=None):
    """
    Generate a list of complete pose sequences, each object is the entire pose sequence for one person
    :return:
    """
    pose_sings_data = []
    pose_sings_meta = []
    for idx in sorted(clip_dict.keys(),):
        sing_pose_np, sing_pose_meta, sing_pose_keys = single_pose_dict2np(clip_dict, idx)
        if scene_id is not None:
            # Format is [scene, clip, index, first frame]
            sing_pose_meta = [int(scene_id), int(clip_id)] + sing_pose_meta
        pose_sings_data.append(sing_pose_np)
        pose_sings_meta += [sing_pose_meta]
    return pose_sings_data, pose_sings_meta


def main():
    """
    A Function for creating an LMDB patch DB out of dataset frame jpeg files.
    Edit paths according to your own.
    :return:
    """
    DATA_ROOT = 'data/'
    vid_dir = {'training': osp.join(DATA_ROOT, 'training/videos/'),
               'testing': osp.join(DATA_ROOT, 'testing/frames/')}

    POSE_ROOT = osp.join(DATA_ROOT, 'pose')
    pose_dir = {'training': osp.join(POSE_ROOT, 'training/tracked_person'),
                'testing': osp.join(POSE_ROOT, 'testing/tracked_person')}

    write_frequency = 100
    patch_size = 16
    for split in ['testing', 'training']:
        patch_size = [patch_size, patch_size]
        extract_patches_2lmdb(pose_dir[split], vid_dir[split],
                              split, patch_size, write_frequency)


if __name__ == '__main__':
    main()
