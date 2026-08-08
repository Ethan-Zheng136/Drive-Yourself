import os
import argparse
import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

def return_front3_cameras(data: wod_e2ed_pb2.E2EDFrame):
    image_list = []
    calibration_list = []
    order = [2, 1, 3]  # front_left, front, front_right

    for camera_name in order:
        for index, image_content in enumerate(data.frame.images):
            if image_content.name == camera_name:
                calibration = data.frame.context.camera_calibrations[index]
                image = tf.io.decode_image(image_content.image).numpy()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                image_list.append(image)
                calibration_list.append(calibration)
                break

    return image_list, calibration_list

def project_vehicle_to_image(vehicle_pose, calibration, points):
  """
  Projects from vehicle coordinate system to image with global shutter.
  """
  # Transform points from vehicle to world coordinate system (can be
  # vectorized).
  pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
  intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
  metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ],
                         dtype=tf.int32)
  camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

  # Perform projection and return projected image coordinates (u, v, ok).
  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()

def draw_points_on_image(image, points, size):
  """
  Draws points on an image.
  """
  for point in points:
    cv2.circle(image, (int(point[0]), int(point[1])), size, (0, 0, 255), -1)
  return image


def visualize_trajectory(
    dataset_path,
    dataset_split,
    token,
    output_path=None,
    point_size=13,
    show_plot=False,
    raw_images_freq=10,
    model_freq=2,
    model_his_frames=9,
    inference_freq=1,
    inference_his_frames=3,
    raw_trajectory_freq=4,
    frame_shift=0,
    scene_frame_interval=1
):
    """
    Visualize trajectory waypoints projected onto front camera image.
    """
    # Lookup LMDB record
    lmdb_path = os.path.join(dataset_path, f'{dataset_split}_lmdb')
    if not os.path.exists(lmdb_path):
        raise ValueError(f"LMDB path does not exist: {lmdb_path}")
    
    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with lmdb_env.begin() as txn:
        raw = txn.get(token.encode('utf-8'))
        if raw is None:
            raise KeyError(f"Token {token} not found in LMDB.")

    # Parse the frame
    frame = wod_e2ed_pb2.E2EDFrame()
    frame.ParseFromString(raw)

    # Extract front camera image and calibration
    front3_images, front3_calibs = return_front3_cameras(frame)
    # front3 order is [front_left, front, front_right]
    front_img = front3_images[1]
    front_calib = front3_calibs[1]

    # Extract future waypoints (x, y, z)
    future_states = frame.future_states
    waypoints = np.stack([
        future_states.pos_x,
        future_states.pos_y,
        future_states.pos_z
    ], axis=1)

    # Project waypoints into image space
    vehicle_pose = frame.frame.images[0].pose
    projected = project_vehicle_to_image(vehicle_pose, front_calib, waypoints)
    points_uv = projected[:, :2]

    # Draw trajectory onto the image
    img_vis = draw_points_on_image(front_img.copy(), points_uv, size=point_size)

    # Save visualization
    if output_path is None:
        output_path = os.path.join('./', f'{token}_front_trajectory.png')
    else:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    cv2.imwrite(output_path, img_vis)
    print(f"Visualization saved to: {output_path}")

    # Display if requested
    if show_plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Front View with Trajectory for Token {token}')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Waymo trajectory waypoints projected onto front camera image')
    parser.add_argument('--dataset_path', required=True, help='Path to the Waymo dataset directory')
    parser.add_argument('--dataset_split', required=True, choices=['training', 'val', 'test'], help='Dataset split (train, val, or test)')
    parser.add_argument('--token', required=True, help='Scene token identifier')
    parser.add_argument('--output_path', default=None, help='Path to save the visualization. If not specified, saves to current directory with token name.')
    parser.add_argument('--point_size', type=int, default=13, help='Size of the trajectory points to draw (default: 13)')
    parser.add_argument('--show_plot', action='store_true', help='Display the plot using matplotlib')
    parser.add_argument('--raw_images_freq', type=int, default=10, help='Frequency of raw images (default: 10)')
    parser.add_argument('--model_freq', type=int, default=2, help='Model frequency (default: 2)')
    parser.add_argument('--model_his_frames', type=int, default=9, help='Number of historical frames for model (default: 9)')
    parser.add_argument('--inference_freq', type=int, default=1, help='Inference frequency (default: 1)')
    parser.add_argument('--inference_his_frames', type=int, default=3, help='Number of historical frames for inference (default: 3)')
    parser.add_argument('--raw_trajectory_freq', type=int, default=4, help='Frequency of raw trajectory (default: 4)')
    parser.add_argument('--frame_shift', type=int, default=0, help='Frame shift offset (default: 0)')
    parser.add_argument('--scene_frame_interval', type=int, default=1, help='Scene frame interval (default: 1)')
    
    args = parser.parse_args()
    
    visualize_trajectory(
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        token=args.token,
        output_path=args.output_path,
        point_size=args.point_size,
        show_plot=args.show_plot,
        raw_images_freq=args.raw_images_freq,
        model_freq=args.model_freq,
        model_his_frames=args.model_his_frames,
        inference_freq=args.inference_freq,
        inference_his_frames=args.inference_his_frames,
        raw_trajectory_freq=args.raw_trajectory_freq,
        frame_shift=args.frame_shift,
        scene_frame_interval=args.scene_frame_interval
    )