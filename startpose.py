#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
import argparse
import subprocess
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, default="rgb",
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser_tracker.add_argument("--pd_m", type=str)
parser_tracker.add_argument("--lm_m", type=str)
parser_tracker.add_argument('-xyz', '--xyz', action="store_true")
parser_tracker.add_argument('-c', '--crop', action="store_true")
parser_tracker.add_argument('--no_smoothing', action="store_true")
parser_tracker.add_argument('-f', '--internal_fps', type=int)
parser_tracker.add_argument('--internal_frame_height', type=int, default=640)
parser_tracker.add_argument('-s', '--stats', action="store_true")
parser_tracker.add_argument('-t', '--trace', action="store_true")
parser_tracker.add_argument('--force_detection', action="store_true")

parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None)
parser_renderer.add_argument("-o","--output")

args = parser.parse_args()

if args.edge:
    from BlazeposeDepthaiEdge import BlazeposeDepthai
else:
    from BlazeposeDepthai import BlazeposeDepthai

tracker = BlazeposeDepthai(
    input_src=args.input,
    pd_model=args.pd_m,
    lm_model=args.lm_m,
    smoothing=not args.no_smoothing,
    xyz=args.xyz,
    crop=args.crop,
    internal_fps=args.internal_fps,
    internal_frame_height=args.internal_frame_height,
    force_detection=args.force_detection,
    stats=args.stats,
    trace=args.trace
)

renderer = BlazeposeRenderer(
    tracker,
    show_3d=args.show_3d,
    output=args.output
)

# ---------------------
# START POSE DETECTION
# ---------------------
start_game = False
start_pose_detected = False
last_pose_time = 0

print("ポーズ待機中... 両手を肩より上にあげてください")

while not start_game:
    frame, body = tracker.next_frame()
    if frame is None:
        break

    if body:
        lm = body.landmarks
        # 両手を肩より上にあげていたら開始
        if lm[15][1] < lm[11][1] and lm[16][1] < lm[12][1]:
            if not start_pose_detected:
                last_pose_time = time.time()
                start_pose_detected = True
            elif time.time() - last_pose_time > 1.5:
                print("ゲームスタート！YOLOへ切り替えます")
                start_game = True
        else:
            start_pose_detected = False

    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        renderer.exit()
        tracker.exit()
        sys.exit(1)

# 終了処理
renderer.exit()
tracker.exit()
sys.exit(0)

# ---------------------
# START YOLO TRACKING
# ---------------------
#subprocess.run(["python3", "tracking_yolo.py"])
