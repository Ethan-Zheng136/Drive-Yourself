import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from models.action_tokenizer import ActionTokenizer
from navsim.agents.autovla_agent import AutoVLAAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

IGNORE_INDEX = -100

class SFTDataset(Dataset):
    def __init__(self, data_config, model_config, processor, using_cot=True):
        data_paths = data_config['json_dataset_path']
        sensor_data_paths = data_config['sensor_data_path']

        # Handle both single path (string/Path) and multiple paths (list)
        if isinstance(data_paths, (str, Path)):
            self.data_paths = [Path(data_paths)]
        else:
            self.data_paths = [Path(path) for path in data_paths]
        
        # Handle sensor_data_path - can be single value or list matching data_paths
        if isinstance(sensor_data_paths, list):
            self.sensor_data_paths = sensor_data_paths
        else:
            # Single value - use for all data paths
            self.sensor_data_paths = [sensor_data_paths] * len(self.data_paths)
        
        # Validate lengths match
        if len(self.sensor_data_paths) != len(self.data_paths):
            raise ValueError(
                f"Number of sensor_data_paths ({len(self.sensor_data_paths)}) must match "
                f"number of json_dataset_paths ({len(self.data_paths)})"
            )
            
        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, model_config=model_config)
        # Flag to control whether to use CoT in training data
        self.using_cot = using_cot
        
        trajectory_sampling = TrajectorySampling(time_horizon=model_config['trajectory']['time_horizon'], 
                                                interval_length=model_config['trajectory']['interval_length'])
        # Use first sensor_data_path for agent init (will be overridden per-scene)
        nuplan_agent = AutoVLAAgent(trajectory_sampling=trajectory_sampling,
                                    sensor_data_path=self.sensor_data_paths[0],
                                   codebook_cache_path=model_config['codebook_cache_path'],
                                   skip_model_load=True)

        self._agent = nuplan_agent
        
        # Get all JSON files from all data paths, tracking which sensor_data_path to use
        # Store as tuples: (scene_path, sensor_data_path)
        self.scenes = []
        for data_path, sensor_path in zip(self.data_paths, self.sensor_data_paths):
            path_scenes = sorted(list(data_path.glob('*.json')))
            for scene in path_scenes:
                self.scenes.append((scene, sensor_path))
            
        if not self.scenes:
            raise ValueError(f"No JSON files found in any of the provided data paths: {self.data_paths}")

        # Initialize any other necessary attributes here

    def __len__(self):
        # return len(self._scene_loader.tokens)
        return len(self.scenes)

    def __getitem__(self, idx):
        # Load data from JSON file
        input_features: Dict[str, torch.Tensor] = {}
        target_trajectory: Dict[str, torch.Tensor] = {}
        
        # Unpack scene path and its corresponding sensor_data_path
        scene_path, sensor_data_path = self.scenes[idx]
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
            
        for builder in self._agent.get_feature_builders():
            input_features.update(builder.compute_features(scene_data))
        for builder in self._agent.get_target_builders():
            target_trajectory.update(builder.compute_targets(scene_data))
        
        # Override sensor_data_path with the correct one for this scene
        input_features['sensor_data_path'] = sensor_data_path
        
        # image sensor
        images = input_features['images']
        camera_images = {}
        
        # List of camera types to load
        camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']
        
        # When sensor_data_path is set, image paths in the json are relative and
        # need to be prefixed with it. When it is null/empty (e.g. nuScenes stores
        # full paths during preprocessing), use the stored path as-is.
        for camera_type in camera_types:
            camera_images[camera_type] = []
            for i in range(4):
                img = images[camera_type][i]
                if input_features['sensor_data_path']:
                    camera_images[camera_type].append(
                        os.path.join(input_features['sensor_data_path'], img))
                else:
                    camera_images[camera_type].append(img)

        # Assign to individual variables for message formatting
        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images['front_camera']
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images['front_left_camera']
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images['front_right_camera']

        # vehicle state
        velocity = input_features["vehicle_velocity"]
        if isinstance(velocity, list) or isinstance(velocity, np.ndarray):
            velocity_x = velocity[0]
            velocity_y = velocity[1]
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
            
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, list) or isinstance(acceleration, np.ndarray):
            acceleration_x = acceleration[0]
            acceleration_y = acceleration[1]
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

        instruction = input_features["driving_command"].lower()

        # trajectory
        gt_action_idx = target_trajectory['gt_idx']
        gt_raw_trajectory = target_trajectory['gt_pos_raw']


        # get assistent content for different datasets
        gt_cot = scene_data['cot_output']

        
        if not self.using_cot:
            assistant_content = [
                {
                    "type": "text",
                    "text": (
                        "<answer>\n"
                        "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                        "</answer>"
                    )
                }
            ]
            system_content = [
                {
                    "type": "text",
                    "text": (
                        "You are an Advanced Driver Assistance and Full Self-Driving System. "
                        "You will be provided with video observations from the ego vehicle's surrounding cameras, along with the vehicle's current dynamic states. "
                        "Your task is to predict the most appropriate driving action for the next five seconds."
                    )
                }
            ]

            has_cot = False
        else:
            system_content = [
                {
                    "type": "text",
                    "text": (
                        "You are an Advanced Driver Assistance and Full Self-Driving System. "
                        "You will receive visual observations from the ego vehicle's cameras and dynamic information about the vehicle's current state. "
                        "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                        "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                        "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                        "Present the final action clearly after your reasoning steps."
                    )
                }
            ]

            # Otherwise, check dataset type and CoT availability
            if scene_data["dataset_name"] == "nuplan" or scene_data["dataset_name"] == "waymo" :
                has_cot = False
                if isinstance(gt_cot, str):
                    assistant_content = [
                        {
                            "type": "text",
                            "text": (
                                "<think>\n"
                                "This is a complex scenario requiring additional reasoning.\n"
                                f"{gt_cot}\n"
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                            )
                        }
                    ]
                    has_cot = True
                else:
                    assistant_content = [
                        {
                            "type": "text",
                            "text": (
                                "<think>\n"
                                "This is a straightforward scenario, and a direct decision can be made.\n"
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                            )
                        }
                    ]
            elif scene_data['dataset_name'] == "nuscenes":
                has_cot = False
                if len(gt_cot) == 5:  # If CoT is available
                    if gt_cot[4] == "STOP\n":
                        gt_cot[4] = "stop"
                    assistant_content = [
                        {
                            "type": "text",
                            "text":
                                "<think>\n"
                                "This is a complex scenario requiring additional reasoning.\n" 
                                f"### Scene Description:\n{gt_cot[0]}\n\n" 
                                f"### Critical Object Description:\n{gt_cot[1] + gt_cot[2]}\n\n" 
                                f"### Reasoning on Intent:\n{gt_cot[3]}\n\n"
                                f"### Best Driving Action:\n{gt_cot[4]}\n" 
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                        }
                    ]
                    has_cot = True
                else:  # Only return the final action
                    assistant_content = [
                        {
                            "type": "text",
                            "text": (
                                "<think>\n"
                                "This is a straightforward scenario, and a direct decision can be made.\n"
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                            )
                        }
                    ]
            else:
                print(scene_data['dataset_name'])
                exit()


        user_content = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."
                )
            },
            {
                "type": "text",
                "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": 28 * 28 * 128,
                "max_pixels": 28 * 28 * 128,
                "video": [
                    f"file://{front_camera_1}",
                    f"file://{front_camera_2}",
                    f"file://{front_camera_3}",
                    f"file://{front_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": 28 * 28 * 128,
                "max_pixels": 28 * 28 * 128,
                "video": [
                    f"file://{front_left_camera_1}",
                    f"file://{front_left_camera_2}",
                    f"file://{front_left_camera_3}",
                    f"file://{front_left_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": 28 * 28 * 128,
                "max_pixels": 28 * 28 * 128,
                "video": [
                    f"file://{front_right_camera_1}",
                    f"file://{front_right_camera_2}",
                    f"file://{front_right_camera_3}",
                    f"file://{front_right_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                    f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
                )
            },
        ]

        # create messages
        messages = [
            {   
                "role": "system",
                "content": system_content
            },

            {
                "role": "user",
                "content": user_content
            },

            # assistant response
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]


        # process the images and messages
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
    
        inputs = {'text': text, 'image_inputs': image_inputs, 'video_inputs': video_inputs}

        # trajectory information
        inputs['gt_trajectory'] = gt_raw_trajectory
        inputs['gt_action'] = gt_action_idx
        inputs['has_cot'] = has_cot
        inputs['data_path'] = scene_path
        # YouDrive: also carry raw features + token so the GRPO joint loss can roll out
        # (generate_sample needs input_features) and score PDMS (needs token), alongside CE.
        inputs['input_features'] = input_features
        inputs['token'] = scene_data['token']
        # Force garbage collection to free temporary objects
        # gc.collect()

  
        return inputs


@dataclass
class DataCollator:
    processor: AutoProcessor
    ignore_index: int = -100
    assistant_id: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.assistant_id is None:
            self.assistant_id = [151644, 77091]  # default value for Qwen2.5-VL

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract text inputs
        text = [batch["text"] for batch in features]
        
        # Process video and image inputs
        video_inputs = []
        image_inputs = []
        has_cot = []
        for batch in features:
            video_inputs.extend(batch["video_inputs"])
            image_inputs.append(batch["image_inputs"])
            has_cot.append(batch["has_cot"])
        
        
        batch = self.processor(
            text=text,
            images=image_inputs if image_inputs[0] is not None else None,
            videos=video_inputs if video_inputs[0] is not None else None,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        # Find the start of the assistant response
        assistant_id = torch.tensor(self.assistant_id)
        for i in range(labels.shape[0]):
            for j in range(len(labels[i]) - len(assistant_id) + 1):
                if torch.equal(labels[i][j:j + len(assistant_id)], assistant_id):
                    start_idx = j
                    break
            
            # [CRITICAL] We take the losses on the assistant response only         
            labels[i, :start_idx] = self.ignore_index
        
        # add labels and gt action to the batch
        batch['labels'] = labels
        batch['gt_trajectory'] = torch.stack([batch['gt_trajectory'] for batch in features])
        batch['gt_action'] = torch.stack([batch['gt_action'] for batch in features])
        batch['has_cot'] = torch.tensor(has_cot)
        # YouDrive: pass through raw features + token for the GRPO rollout (batch_size=1).
        if 'input_features' in features[0]:
            batch['input_features'] = features[0]['input_features']
            batch['token'] = features[0]['token']

        return batch