from typing import List, Optional
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
import tqdm


def create_dataset_from_scenes(args) -> None:
    dataframe = create_dataframe(args)
    scenes_path = os.path.join(os.getcwd(), args.input_dir, '*.*')
    for scene in tqdm.tqdm(glob.glob(scenes_path)):
        cap = cv2.VideoCapture(scene)
        frame_group_idx = 0
        while True:
            frame_groups = get_frame_groups(cap, args.num_frames, args.frame_rate)
            if frame_groups is None:
                break
            for frame_group in frame_groups:
                frame_group_paths = save_frame_group(frame_group, frame_group_idx, args.output_dir, scene)
                frame_group_idx += 1
                dataframe = add_frame_group_paths(dataframe, frame_group_paths)
        cap.release()
    save_dataframe(dataframe, args.output_dir, args.dataset_file_name)


def create_dataframe(args) -> pd.DataFrame:
    columns = [f'frame{int(idx)}' for idx in range(args.num_frames)]
    dataframe = pd.DataFrame(columns=columns)
    return dataframe


def get_frame_groups(video_capture: cv2.VideoCapture,
                     num_frames: int,
                     frame_rate: int) -> Optional[List[List[np.ndarray]]]:
    frame_groups = [list() for _ in range(frame_rate)]
    for idx in range(num_frames * frame_rate):
        ret, frame = video_capture.read()
        if not ret:
            return None
        frame_groups[int(idx % frame_rate)].append(frame)
    return frame_groups


def save_frame_group(frames: List[np.ndarray], frame_group_idx: int, output_dir: str, input_file: str) -> List[str]:
    filename = os.path.splitext(os.path.split(input_file)[1])[0]
    path_to_frame = {f'{filename}__group_{frame_group_idx}__frame_{idx}.png': frame
                     for idx, frame in enumerate(frames)}
    paths = []
    for path, frame in path_to_frame.items():
        path = os.path.join(output_dir, path)
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


def add_frame_group_paths(dataframe: pd.DataFrame, frame_group_paths: List[str]) -> pd.DataFrame:
    row = {f'frame{idx}': frame_group_path for idx, frame_group_path in enumerate(frame_group_paths)}
    dataframe = dataframe.append(row, ignore_index=True)
    return dataframe


def save_dataframe(dataframe: pd.DataFrame, output_dir: str, dataset_file_name: str) -> None:
    output_path = os.path.join(output_dir, dataset_file_name)
    dataframe.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dir', '-i', type=str, required=True, help='Directory where input scene files are located'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, required=True, help='Directory where to save output frames'
    )
    parser.add_argument(
        '--dataset-file-name', '-dfm', type=str, required=True,
        help='Name of the output dataset csv file describing each group of frames'
    )
    parser.add_argument(
        '--num-frames', '-nf', type=int, required=True,
        help='Number of frames in a single group'
    )
    parser.add_argument(
        '--frame-rate', '-fr', type=int, required=True,
        help='Frequency at which frames are sampled'
    )

    args = parser.parse_args()
    create_dataset_from_scenes(args)
