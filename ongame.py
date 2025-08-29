#!/usr/bin/env python3
import argparse
import cv2
import time
import sys
import threading
from playsound import playsound

#AkariClientのインポート
from akari_client import AkariClient
from akari_client.color import Colors
from akari_client.config import (
    AkariClientConfig,
    JointManagerGrpcConfig,
    M5StackGrpcConfig,
)
from akari_client.position import Positions
from lib.akari_yolo_lib.oakd_tracking_yolo import OakdTrackingYolo

#AkariClient、jointsのインスタンスを取得
akari = AkariClient()
joints = akari.joints
m5 = akari.m5stack

# ファイル名をまとめて定義
AUDIO_FILES = {
    "1": "voicefile/voice01.wav",
    "move":"voicefile/voice_move.wav",
    "start":"voicefile/voice_start.wav",
    "finish":"voicefile/voice_finish.wav"
}

# 音声再生用の関数を宣言
def play_audio_async(key):
    def _play():
        playsound(AUDIO_FILES[key])
    threading.Thread(target=_play, daemon=True).start()
    
def play_audio_sync(key):
    playsound(AUDIO_FILES[key])

def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Provide model name or model path for inference",
        default="yolov7tiny_coco_416x416",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Provide config path for inference",
        default="json/yolov7tiny_coco_416x416.json",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fps",
        help="Camera frame fps. This should be smaller than nn inference fps",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--display_camera",
        help="Display camera rgb and depth frame",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--robot_coordinate",
        help="Convert object pos from camera coordinate to robot coordinate",
        action="store_true",
    )
    parser.add_argument(
        "--spatial_frame",
        help="Display spatial frame instead of bird frame",
        action="store_true",
    )
    parser.add_argument(
        "--disable_orbit",
        help="Disable display tracked orbit on bird frame",
        action="store_true",
    )
    parser.add_argument(
        "--log_path",
        help="Path to save orbit data",
        type=str,
    )
    parser.add_argument(
        "--log_continue",
        help="Continue log data",
        action="store_true",
    )
    args = parser.parse_args()
    bird_frame = False
    orbit = True
    spatial_frame = False
    if args.disable_orbit:
        orbit = False
    # spatial_frameを有効化した場合、bird_frameは無効化
    if args.spatial_frame:
        bird_frame = False
        spatial_frame = True
        orbit = False
    end = False

    #過去位置の保存用
    previous_positions = {}
    #ループ前に時間記録用変数を初期化
    last_move_time = time.time()
    operating_time = 4.0
    waiting_time = 10.0
    move_interval = waiting_time
    
    joints.set_servo_enabled(pan=True, tilt=True)
    toggle = False
    is_detecting = False
    detection_delay = 1.0 
    detection_start_time = None
    OnGame = True
    User_Moved = False
    
    m5.set_display_text(text="ゲーム開始", text_color=Colors.GREEN)
    play_audio_sync("start")

    while not end:
        oakd_tracking_yolo = OakdTrackingYolo(
            config_path=args.config,
            model_path=args.model,
            fps=args.fps,
            cam_debug=args.display_camera,
            robot_coordinate=args.robot_coordinate,
            show_bird_frame=bird_frame,
            show_spatial_frame=spatial_frame,
            show_orbit=orbit,
            log_path=args.log_path,
            log_continue=args.log_continue
        )
        oakd_tracking_yolo.update_bird_frame_distance(10000)
        while True:
            frame = None
            detections = []
            data = m5.get()
            current_time = time.time()
            
            try:
                frame, detections, tracklets = oakd_tracking_yolo.get_frame()
            except BaseException:
                print("===================")
                print("get_frame() error! Reboot OAK-D.")
                print("If reboot occur frequently, Bandwidth may be too much.")
                print("Please lower FPS.")
                print("==================")
                break
                
            if(data["button_c"]==True):
                m5.set_display_text(text="ゲーム終了", text_color=Colors.GREEN)
                play_audio_sync("finish")
                joints.move_joint_positions(pan=0, tilt=0.1)
                OnGame = False
                oakd_tracking_yolo.close()
                sys.exit(0)  # 親プロセスに「終了」を通知


            if OnGame is True:
                #首の動き
                if current_time - last_move_time > move_interval:
                    if toggle:
                        play_audio_sync("1")
                        joints.move_joint_positions(pan=0, tilt=0.1)
                        detection_start_time = time.time() + detection_delay
                        m5.set_display_text(text="検知中", text_color=Colors.GREEN)
                        is_detecting = True
                        move_interval = operating_time
                    else:
                        joints.move_joint_positions(pan=0, tilt=-0.3)
                        detection_start_time = None
                        m5.set_display_text(text="停止中", text_color=Colors.BLUE)
                        is_detecting = False
                        move_interval = waiting_time
                        User_Moved = False
                    toggle = not toggle
                    last_move_time = current_time
            #動きの検出
            if tracklets is not None:
                for tracklet in tracklets:
                    if tracklet.status.name == "TRACKED" and tracklet.label == 0:#jsonのlabels内でparsonは０番目
                        id = tracklet.id
                        x = tracklet.spatialCoordinates.x
                        y = tracklet.spatialCoordinates.y
                        z = tracklet.spatialCoordinates.z
                        
                        if detection_start_time is not None and current_time >= detection_start_time and id in previous_positions:
                            prev_x, prev_y, prev_z = previous_positions[id]
                            dx = abs(x - prev_x)
                            dy = abs(y - prev_y)
                            dz = abs(z - prev_z)
                            if dx > 50 or dy > 50 or dz > 50:
                                if is_detecting and not User_Moved:
                                    print(f"[WARNING] Person {id} moved! Δx={dx}, Δy={dy}, Δz={dz}")
                                    play_audio_async("move")
                                    User_Moved = True
                                    m5.set_display_text(text="動いた！", text_color=Colors.RED)
                        previous_positions[id] = (x, y, z)
            if frame is not None:
                oakd_tracking_yolo.display_frame("nn", frame, tracklets)
            if cv2.waitKey(1) == ord("q"):
                end = True
                joints.move_joint_positions(pan=0, tilt=0.1)
                break
        oakd_tracking_yolo.close()
        sys.exit(1)



if __name__ == "__main__":
    main()
