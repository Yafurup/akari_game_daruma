import subprocess

while True:
    # ポーズ検出フェーズ（終了時のコードで制御）
    result = subprocess.run(["python3", "startpose.py"])
    if result.returncode == 1:
        break  # 終了指示

    # YOLOフェーズ（終了時のコードで制御）
    result = subprocess.run(["python3", "ongame.py"])
    if result.returncode == 1:
        break  # 終了指示
