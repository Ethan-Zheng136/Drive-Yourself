import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import argparse
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
from tqdm import tqdm 
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Transfer Waymo E2E dataset format')
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help='Path to the Waymo E2E dataset folder')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the output folder for images')
    parser.add_argument('--split', type=str, required=True, choices=['training', 'val', 'test'],
                        help='Dataset split: training, val, or test')
    
    args = parser.parse_args()
    
    DATASET_FOLDER = args.dataset_folder
    OUTPUT_FOLDER = args.output_folder
    
    # Select file pattern based on split
    if args.split == 'training':
        FILE_PATTERN = os.path.join(DATASET_FOLDER, 'training_*.tfrecord-*')
    elif args.split == 'val':
        FILE_PATTERN = os.path.join(DATASET_FOLDER, 'val_*.tfrecord-*')
    else: 
        FILE_PATTERN = os.path.join(DATASET_FOLDER, 'test_*.tfrecord-*')
    
    filenames = tf.io.matching_files(FILE_PATTERN)
    dataset = tf.data.TFRecordDataset(filenames, compression_type='')
    dataset_iter = dataset.as_numpy_iterator()
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    camera_mapping = {
        1: 'front',
        2: 'front_left',
        3: 'front_right',
        4: 'left',
        5: 'right',
        6: 'back_left',
        7: 'back',
        8: 'back_right'
    }
    
    sequence_list = []
    
    for idx, bytes_example in tqdm(enumerate(dataset_iter), desc="Processing frames"):
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(bytes_example)
    
        sequence_name = data.frame.context.name.split('-')[0]
        for image_content in data.frame.images:
            
            # target folder and create the folder
            camera_name = camera_mapping[image_content.name]
            folder_name = os.path.join(OUTPUT_FOLDER, sequence_name, camera_name)
            os.makedirs(folder_name, exist_ok=True)
    
            image = tf.io.decode_image(image_content.image).numpy()
            img_filename = f"{data.frame.context.name}.jpg"
            img_path = os.path.join(folder_name, img_filename)
            Image.fromarray(image).save(img_path, format='JPEG')
    
    print(f"Successfully transferred the Waymo E2E data, and save it in {OUTPUT_FOLDER}")

if __name__ == '__main__':
    main()