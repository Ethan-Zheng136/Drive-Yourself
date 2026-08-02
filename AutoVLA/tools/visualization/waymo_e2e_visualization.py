import os
import json
import textwrap
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def wrap_text(text, width):
    """
    Wrap text to the specified width while preserving existing newline characters.

    Args:
      text (str): The original text string.
      width (int): The maximum number of characters per line.
    
    Returns:
      str: The wrapped text.
    """
    wrapped_lines = []
    for line in text.splitlines():
        wrapped_lines.append(textwrap.fill(line, width=width))
    return "\n".join(wrapped_lines)


def save_debug_visualization(
    scene_folder,
    token,
    sample,
    gt_trajectory,
    his_trajectory,
    cot_text=None,
    include_cot=True
):
    """
    Save a debug visualization image that includes a 9-grid camera view,
    a BEV trajectory plot, and optionally CoT output text.

    Args:
      scene_folder (str): The folder path for storing scene results.
      token (str): The scene identifier token.
      sample (dict): The sample data containing image path information.
      gt_trajectory (list): List of target trajectory points.
      his_trajectory (list): List of historical trajectory points.
      cot_text (str): The CoT inference output text, including formatted Markdown.
      include_cot (bool): flag for the CoT data
    """
    os.makedirs(scene_folder, exist_ok=True)

    # Choose layout depending on include_cot
    if include_cot:
        n_rows = 4
        height_ratios = [1, 1, 1, 0.4]
    else:
        n_rows = 3
        height_ratios = [1, 1, 1]

    fig = plt.figure(figsize=(12, 12 if not include_cot else 15))
    gs = gridspec.GridSpec(n_rows, 3, height_ratios=height_ratios, hspace=0.2, wspace=0.1)

    camera_positions = {
        (0, 0): "front_left_camera_paths",
        (0, 1): "front_camera_paths",
        (0, 2): "front_right_camera_paths",
        (1, 0): "left_camera_paths",
        (1, 2): "right_camera_paths",
        (2, 0): "back_left_camera_paths",
        (2, 1): "back_camera_paths",
        (2, 2): "back_right_camera_paths"
    }

    # Plot camera views
    for (r, c), key in camera_positions.items():
        ax = plt.subplot(gs[r, c])
        paths = sample.get(key, [])
        if paths:
            img_path = paths[-1]
            if img_path and os.path.isfile(img_path):
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(key.replace('_camera_paths', ''))
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "No Image", ha='center', va='center')
        ax.axis('off')

    # Plot BEV trajectory
    ax_traj = plt.subplot(gs[1, 1])
    if his_trajectory:
        x_past = [pt[0] for pt in his_trajectory]
        y_past = [pt[1] for pt in his_trajectory]
        ax_traj.plot(x_past, y_past, 'bo-', label='Past Trajectory')
    if gt_trajectory:
        x_future = [pt[0] for pt in gt_trajectory]
        y_future = [pt[1] for pt in gt_trajectory]
        ax_traj.plot(x_future, y_future, 'ro-', label='Future Trajectory')
    ax_traj.set_title(sample.get('fut_ego_action', 'No Action'))
    ax_traj.set_xlabel('X')
    ax_traj.set_ylabel('Y')
    ax_traj.legend()
    ax_traj.grid(True)
    ax_traj.axis('equal')

    # Optionally add CoT text
    if include_cot and cot_text:
        wrapped = wrap_text(cot_text, width=100)
        ax_text = plt.subplot(gs[3, :])
        ax_text.axis('off')
        ax_text.text(0, 1, wrapped, fontsize=11, fontfamily='monospace', va='top', transform=ax_text.transAxes)

    # Save image
    out_path = os.path.join(scene_folder, f"{token}_debug.jpg")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def process_file(json_file, json_folder, output_folder, include_cot=True):
    json_path = os.path.join(json_folder, json_file)
    try:
        with open(json_path) as f:
            sample = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

    token = sample.get('token', os.path.splitext(json_file)[0])
    gt_traj = sample.get('gt_trajectory', [])
    his_traj = sample.get('his_trajectory', [])
    cot = sample.get('cot_output', '') if include_cot else None

    save_debug_visualization(
        output_folder,
        token,
        sample,
        gt_traj,
        his_traj,
        cot_text=cot,
        include_cot=include_cot
    )
    return token


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Generate debug visualizations with optional CoT text'
    )
    parser.add_argument('--json_folder', required=True, help='Path to JSON folder')
    parser.add_argument('--output_folder', required=True, help='Path to save visualizations')
    parser.add_argument('--include_cot', action='store_true', help='Include CoT output text')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    files = [f for f in os.listdir(args.json_folder) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files.")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_file, f, args.json_folder, args.output_folder, args.include_cot): f
            for f in files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing JSON files'):
            f = futures[future]
            try:
                token = future.result()
                if token:
                    print(f"Processed token: {token}")
            except Exception as exc:
                print(f"{f} exception: {exc}")