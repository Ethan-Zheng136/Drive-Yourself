"""
NuScenes Dataset Preprocessing Script for DriveVLA.

This script extracts trajectory data and camera paths from the NuScenes dataset
and saves them as JSON files for training and validation.

Usage:
    python tools/preprocessing/nusc_sample_generation.py \
        --nuscenes_path /path/to/nuscenes \
        --output_dir /path/to/output \
        --split train \
        --drivelm_path /path/to/drivelm/v1_1_train_nus.json

Arguments:
    --nuscenes_path: Path to NuScenes dataset root directory
    --output_dir: Output directory for preprocessed JSON files
    --split: Dataset split to process (train or val)
    --version: NuScenes dataset version (default: v1.0-trainval)
    --drivelm_path: Optional path to DriveLM annotations JSON file
"""

import json
import os
import argparse
import numpy as np
import torch
import math

from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils import splits


def convert_to_json_serializable(obj):
    """Convert numpy arrays and torch tensors to JSON serializable format."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)


def quart_to_rpy(qua):
    """Convert quaternion to roll, pitch, yaw."""
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def get_global_sensor_pose(rec, nusc, inverse=False):
    """Get global sensor pose transformation matrix."""
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])

    if not inverse:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)

    return pose


def get_ego_pose_future_his(nusc, fut_ts, sample, his_ts):
    """Extract ego trajectory for history and future frames."""
    ego_his_trajs = np.zeros((his_ts + 1, 3))
    ego_his_trajs_diff = np.zeros((his_ts + 1, 3))
    ego_his_masks = np.zeros((his_ts + 1))
    sample_cur = sample
    
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    
    # Extract history trajectory
    for i in range(his_ts, -1, -1):
        if sample_cur is not None:
            pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
            ego_his_trajs[i] = pose_mat[:3, 3]
            ego_his_masks[i] = 1
            has_prev = sample_cur['prev'] != ''
            has_next = sample_cur['next'] != ''
            if has_next:
                sample_next = nusc.get('sample', sample_cur['next'])
                pose_mat_next = get_global_sensor_pose(sample_next, nusc, inverse=False)
                ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]
            sample_cur = nusc.get('sample', sample_cur['prev']) if has_prev else None
        else:
            ego_his_trajs[i] = ego_his_trajs[i + 1] - ego_his_trajs_diff[i + 1]
            ego_his_trajs_diff[i] = ego_his_trajs_diff[i + 1]
    
    # Transform to ego frame (global to ego at lcf)
    ego_his_trajs = ego_his_trajs - np.array(pose_record['translation'])
    rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
    ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
    
    # Ego to lidar at lcf
    ego_his_trajs = ego_his_trajs - np.array(cs_record['translation'])
    rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
    ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
    ego_his_diff = ego_his_trajs[1:] - ego_his_trajs[:-1]

    # Extract future trajectory
    ego_fut_trajs = np.zeros((fut_ts + 1, 3))
    ego_fut_masks = np.zeros((fut_ts + 1))
    sample_cur = sample
    
    for i in range(fut_ts + 1):
        pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
        ego_fut_trajs[i] = pose_mat[:3, 3]
        ego_fut_masks[i] = 1
        if sample_cur['next'] == '':
            ego_fut_trajs[i + 1:] = ego_fut_trajs[i]
            break
        else:
            sample_cur = nusc.get('sample', sample_cur['next'])
    
    # Transform to ego frame
    ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
    rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
    rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    
    # Determine driving command based on final future position
    if ego_fut_trajs[-1][0] >= 2:
        command = 'Turn Right'
    elif ego_fut_trajs[-1][0] <= -2:
        command = 'Turn Left'
    else:
        command = 'Go Straight'
    
    ego_fut_diff = ego_fut_trajs[1:] - ego_fut_trajs[:-1]
    
    return (
        ego_fut_diff[:, :2].astype(np.float32),
        ego_fut_trajs[:, :2].astype(np.float32),
        ego_his_trajs[:, :2].astype(np.float32),
        ego_his_diff[:, :2].astype(np.float32),
        ego_fut_masks[1:].astype(np.float32),
        ego_his_masks[:-1].astype(np.float32),
        command
    )


def get_ego_velocity(sample, nusc):
    """Calculate ego vehicle velocity from consecutive frames."""
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

    if sample['prev'] != '':
        sample_prev = nusc.get('sample', sample['prev'])
        sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
        pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
    else:
        pose_record_prev = None

    assert (pose_record_prev is not None), 'prev token is empty'

    ego_pos = np.array(pose_record['translation'])
    ego_pos_prev = np.array(pose_record_prev['translation'])
    
    # Velocity = distance / time (0.5s between frames at 2Hz)
    ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
    return ego_v


def get_planning_instruction(ego_fut_diff, ego_fut_trajs, ego_his_diff, ego_his_trajs):
    """Generate planning instruction based on trajectory analysis."""
    meta_action = ""

    # speed meta
    constant_eps = 0.5
    his_velos = np.linalg.norm(ego_his_diff, axis=1)
    fut_velos = np.linalg.norm(ego_fut_diff, axis=1)
    cur_velo = his_velos[-1]
    end_velo = fut_velos[-1]

    if cur_velo < constant_eps and end_velo < constant_eps:
        speed_meta = "stop"
    elif end_velo < constant_eps:
        speed_meta = "a deceleration to zero"
    elif np.abs(end_velo - cur_velo) < constant_eps:
        speed_meta = "a constant speed"
    else:
        if cur_velo > end_velo:
            if cur_velo > 2 * end_velo:
                speed_meta = "a quick deceleration"
            else:
                speed_meta = "a deceleration"
        else:
            if end_velo > 2 * cur_velo:
                speed_meta = "a quick acceleration"
            else:
                speed_meta = "an acceleration"
    
    # behavior_meta
    if speed_meta == "stop":
        meta_action += (speed_meta + "\n")
        return meta_action.upper()
    else:
        forward_th = 2.0
        lane_changing_th = 4.0
        if (np.abs(ego_fut_trajs[:, 0]) < forward_th).all():
            behavior_meta = "move forward"
        else:
            if ego_fut_trajs[-1, 0] < 0:  # left
                if np.abs(ego_fut_trajs[-1, 0]) > lane_changing_th:
                    behavior_meta = "turn left"
                else:
                    behavior_meta = "chane lane to left"
            elif ego_fut_trajs[-1, 0] > 0:  # right
                if np.abs(ego_fut_trajs[-1, 0]) > lane_changing_th:
                    behavior_meta = "turn right"
                else:
                    behavior_meta = "change lane to right"
            else:
                raise ValueError(f"Undefined behaviors: {ego_fut_trajs}")
                
        meta_action += (behavior_meta + " with " + speed_meta)
        return meta_action


def extract_nuscenes_data(nusc, save_path, split='train', drivelm_path=None):
    """
    Extract data from NuScenes dataset.
    
    Args:
        nusc: NuScenes instance
        save_path: Output directory for JSON files
        split: 'train' or 'val'
        drivelm_path: Optional path to DriveLM annotations
    """
    # Get scene splits
    train_scenes = splits.train
    val_scenes = splits.val

    available_scenes = nusc.scene
    available_scene_names = [s['name'] for s in available_scenes]

    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    
    train_scene_tokens = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scene_tokens = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    scene_set = train_scene_tokens if split == 'train' else val_scene_tokens
    
    # Load DriveLM data if provided
    drivelm_samples = set()
    if drivelm_path and os.path.exists(drivelm_path):
        print(f"Loading DriveLM annotations from {drivelm_path}")
        with open(drivelm_path, 'r') as f:
            drivelm_data = json.load(f)
        drivelm_samples = process_drivelm_data(
            nusc, drivelm_data, scene_set, save_path
        )
    
    # Process raw NuScenes samples
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
    fut_ts = 16
    his_ts = 3
    
    processed_count = 0
    skipped_count = 0
    
    for sample in tqdm(nusc.sample, desc=f"Processing NuScenes {split} samples"):
        # Skip if already processed from DriveLM
        if sample['token'] in drivelm_samples:
            continue
        
        # Filter by scene
        if sample['scene_token'] not in scene_set:
            continue
        
        # Skip if no previous frame (needed for velocity calculation)
        if sample['prev'] == '':
            skipped_count += 1
            continue
        
        frame_id = sample['token']
        sample_data = {'token': frame_id, 'dataset_name': 'nuscenes'}
        
        # Get trajectory data
        result = get_ego_pose_future_his(nusc, fut_ts, sample, his_ts)
        gt_ego_fut_diff, gt_ego_fut_trajs, gt_ego_his_trajs, gt_ego_his_diff, \
            gt_ego_fut_masks, gt_ego_his_masks, command = result
        
        # Validate trajectory data (only check future masks for training)
        if split == 'train':
            if np.sum(gt_ego_fut_masks) < 10:
                skipped_count += 1
                continue
        if np.sum(gt_ego_his_masks) < 3:
            skipped_count += 1
            continue

        # Get velocity and acceleration
        ego_v = get_ego_velocity(sample, nusc)
        sample_prev = nusc.get('sample', sample['prev'])
        ego_v_previous = get_ego_velocity(sample_prev, nusc)
        ego_acc = (ego_v - ego_v_previous) / 0.5

        # Get camera paths for history frames
        sample_data['front_camera_paths'] = []
        sample_data['front_right_camera_paths'] = []
        sample_data['front_left_camera_paths'] = []
        sample_data['back_camera_paths'] = []
        sample_data['back_right_camera_paths'] = []
        sample_data['back_left_camera_paths'] = []

        sample_cur = sample
        for i in range(his_ts, -1, -1):
            if sample_cur is not None:
                for camera_type in camera_types:
                    key_suffix = camera_type.lower().replace('cam_', '')
                    cam_token = sample_cur['data'][camera_type]
                    cam_path, _, _ = nusc.get_sample_data(cam_token)
                    sample_data[f'{key_suffix}_camera_paths'].insert(0, cam_path)
                if i != 0:
                    sample_cur = nusc.get('sample', sample_cur['prev'])
            else:
                skipped_count += 1
                break
        else:
            # Convert trajectory to x-forward, y-left coordinate frame
            gt_ego_fut_trajs_output = gt_ego_fut_trajs[:11]
            gt_ego_fut_trajs_output_f = np.zeros((gt_ego_fut_trajs_output.shape[0], 3))
            gt_ego_fut_trajs_output_f[:, 0] = gt_ego_fut_trajs_output[:, 1]
            gt_ego_fut_trajs_output_f[:, 1] = -gt_ego_fut_trajs_output[:, 0]

            # Calculate heading angles
            heading = np.arctan2(
                gt_ego_fut_trajs_output_f[1:, 1] - gt_ego_fut_trajs_output_f[:-1, 1],
                gt_ego_fut_trajs_output_f[1:, 0] - gt_ego_fut_trajs_output_f[:-1, 0] + 1e-3
            )
            gt_ego_fut_trajs_output_f[1:, 2] = heading

            sample_data['gt_trajectory'] = gt_ego_fut_trajs_output_f[1:]
            sample_data['cot_output'] = []
            sample_data['instruction'] = command
            sample_data['velocity'] = ego_v
            sample_data['acceleration'] = ego_acc
            
            # Include future_mask for evaluation data
            if split == 'val':
                sample_data['future_mask'] = gt_ego_fut_masks[:10]

            # Save to JSON
            json_data = convert_to_json_serializable(sample_data)
            output_path = os.path.join(save_path, f"{frame_id}.json")
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            processed_count += 1
    
    print(f"Processed {processed_count} samples, skipped {skipped_count} samples")
    return processed_count


def process_drivelm_data(nusc, drivelm_data, scene_set, save_path):
    """Process samples with DriveLM annotations."""
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
    fut_ts = 16
    his_ts = 3
    processed_samples = set()
    
    for scene_id in tqdm(drivelm_data.keys(), desc="Processing DriveLM data"):
        if scene_id not in scene_set:
            continue
            
        scene_data = drivelm_data[scene_id]['key_frames']
        
        for frame_id in scene_data.keys():
            processed_samples.add(frame_id)
            frame_data_infos = scene_data[frame_id]['key_object_infos']
            frame_data_qa = scene_data[frame_id]['QA']
            
            sample_data = {'token': frame_id, 'dataset_name': 'nuscenes'}
            
            # Extract QA data
            qa_data = {
                'perception': [],
                'prediction': [],
                'fov': [],
                'move_intent': []
            }

            perception = frame_data_qa.get("perception", [])
            prediction = frame_data_qa.get("prediction", [])

            # Process perception QA
            for qa in perception:
                question = qa['Q'].lower()
                answer = qa['A']
                
                fov_questions = [
                    'what are objects to the front of the ego car?',
                    'what are objects to the back of the ego car?',
                    'what are objects to the front right of the ego car?',
                    'what are objects to the front left of the ego car?',
                    'what are objects to the back left of the ego car?',
                    'what are objects to the back right of the ego car?'
                ]
                
                for fov_q in fov_questions:
                    if fov_q in question:
                        qa_data['fov'].append(answer)
                        break
                
                if 'what are the important objects in the current scene?' in question:
                    description = answer.split('.')[0]
                    qa_data['perception'].append(description)

            # Process prediction QA
            for qa in prediction:
                question = qa['Q']
                answer = qa['A']
                
                if "what object should the ego vehicle notice first" in question.lower():
                    description = answer
                    while '<' in description and '>' in description:
                        obj_id = description[description.find('<'):description.find('>') + 1]
                        obj_id_nospace = obj_id.replace(" ", "")
                        if obj_id_nospace in frame_data_infos:
                            obj_description = "**" + frame_data_infos[obj_id_nospace]['Visual_description'].split('.')[0] + "**"
                            description = description.replace(obj_id, obj_description)
                        else:
                            break
                    qa_data['prediction'].append(description)

                if "what is the future state" in question.lower():
                    move_intent = answer.split('.')[0].lower()
                    move_obj = question[question.find('<'):question.find('>') + 1]
                    move_obj_nospace = move_obj.replace(" ", "")
                    if move_obj_nospace in frame_data_infos:
                        move_obj_description = "**" + frame_data_infos[move_obj_nospace]['Visual_description'].split('.')[0] + "**"
                        move_status = f"The moving status of {move_obj_description} is {move_intent}."
                        qa_data['move_intent'].append(move_status)

            # Get trajectory data
            try:
                sample = nusc.get('sample', frame_id)
            except KeyError:
                continue
                
            result = get_ego_pose_future_his(nusc, fut_ts, sample, his_ts)
            gt_ego_fut_diff, gt_ego_fut_trajs, gt_ego_his_trajs, gt_ego_his_diff, \
                gt_ego_fut_masks, gt_ego_his_masks, command = result
            
            if np.sum(gt_ego_fut_masks) < 10 or np.sum(gt_ego_his_masks) < 3:
                continue

            ego_v = get_ego_velocity(sample, nusc)
            if sample['prev'] == '':
                continue
            sample_prev = nusc.get('sample', sample['prev'])
            ego_v_previous = get_ego_velocity(sample_prev, nusc)
            ego_acc = (ego_v - ego_v_previous) / 0.5

            # Get camera paths
            sample_data['front_camera_paths'] = []
            sample_data['front_right_camera_paths'] = []
            sample_data['front_left_camera_paths'] = []
            sample_data['back_camera_paths'] = []
            sample_data['back_right_camera_paths'] = []
            sample_data['back_left_camera_paths'] = []

            sample_cur = sample
            valid = True
            for i in range(his_ts, -1, -1):
                if sample_cur is not None:
                    for camera_type in camera_types:
                        key_suffix = camera_type.lower().replace('cam_', '')
                        cam_token = sample_cur['data'][camera_type]
                        cam_path, _, _ = nusc.get_sample_data(cam_token)
                        sample_data[f'{key_suffix}_camera_paths'].insert(0, cam_path)
                    if i != 0:
                        sample_cur = nusc.get('sample', sample_cur['prev'])
                else:
                    valid = False
                    break
            
            if not valid:
                continue

            # Convert trajectory
            gt_ego_fut_trajs_output = gt_ego_fut_trajs[:11]
            gt_ego_fut_trajs_output_f = np.zeros((gt_ego_fut_trajs_output.shape[0], 3))
            gt_ego_fut_trajs_output_f[:, 0] = gt_ego_fut_trajs_output[:, 1]
            gt_ego_fut_trajs_output_f[:, 1] = -gt_ego_fut_trajs_output[:, 0]

            heading = np.arctan2(
                gt_ego_fut_trajs_output_f[1:, 1] - gt_ego_fut_trajs_output_f[:-1, 1],
                gt_ego_fut_trajs_output_f[1:, 0] - gt_ego_fut_trajs_output_f[:-1, 0] + 1e-3
            )
            gt_ego_fut_trajs_output_f[1:, 2] = heading

            sample_data['gt_trajectory'] = gt_ego_fut_trajs_output_f[1:]
            sample_data['instruction'] = command
            sample_data['velocity'] = ego_v
            sample_data['acceleration'] = ego_acc

            # Get planning instruction
            planning_instruction = get_planning_instruction(
                gt_ego_fut_diff, gt_ego_fut_trajs, gt_ego_his_diff, gt_ego_his_trajs
            )

            # Build CoT output
            fov_text = ' '.join(qa_data['fov']) if qa_data['fov'] else ''
            perception_text = qa_data['perception'][0] if qa_data['perception'] else ''
            move_intent_text = ' '.join(qa_data['move_intent']) if qa_data['move_intent'] else ''
            prediction_text = qa_data['prediction'][0] if qa_data['prediction'] else ''
            
            sample_data['cot_output'] = [fov_text, perception_text, move_intent_text, prediction_text, planning_instruction]

            # Save to JSON
            json_data = convert_to_json_serializable(sample_data)
            output_path = os.path.join(save_path, f"{frame_id}.json")
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
    
    return processed_samples


def main():
    parser = argparse.ArgumentParser(
        description="NuScenes Dataset Preprocessing for DriveVLA"
    )
    parser.add_argument(
        "--nuscenes_path", type=str, required=True,
        help="Path to NuScenes dataset root directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for preprocessed JSON files"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"],
        help="Dataset split to process (train or val)"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-trainval",
        help="NuScenes dataset version"
    )
    parser.add_argument(
        "--drivelm_path", type=str, default=None,
        help="Optional path to DriveLM annotations JSON file"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize NuScenes
    print(f"Loading NuScenes {args.version} from {args.nuscenes_path}")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_path, verbose=True)

    # Process data
    print(f"Processing {args.split} split...")
    extract_nuscenes_data(
        nusc=nusc,
        save_path=args.output_dir,
        split=args.split,
        drivelm_path=args.drivelm_path
    )

    print(f"Done! Preprocessed data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
