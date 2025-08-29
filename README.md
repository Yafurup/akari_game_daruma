
# akari_game_daruma
AKARIとだるまさんがころんだで遊べるアプリです

## 概要
AKARIにだるまさんがころんだの鬼をやってもらうアプリ.  
・両手を肩より上に挙げることでゲームが開始される.  
・AKARIが正面を向いている時に、動いた人がいるとディスプレイと音声で伝える.  
・AKARIのディスプレイ右下のボタンを押すことでゲーム終了.  

## セットアップ手順
1. ローカルにクローンする  
cd ~  
git clone　https://github.com/AkariGroup/akari_game_daruma   
cd akari_game_daruma  
2. submoduleの更新  
git submodule update --init  
3. 仮想環境の作成  
python3 -m venv venv  
. venv/bin/activate　　
sudo apt install portaudio19-dev　　
pip install -r requirements.txt  

## 起動方法
1. 仮想環境の有効化    
. venv/bin/activate  
2. プログラムの起動  
python3 main.py  
3. 終了する時
起動しているウィンドウを選択した状態でキーボードのqキーを押す

## 使い方
1. main.pyを実行します
2. ポーズを検出するプログラムが起動します  
3. 両手を肩より上に挙げることでゲームを開始します
4. AKARIが正面を向いている間は動かないように近づいてください
5. ディスプレイ右下のボタンを押すことでゲームを終了します
6. 2.に戻ります
7. プログラム自体を終了する場合は起動しているウィンドウを選択した状態でqキーを入力してください

## その他
このアプリケーションは愛知工業大学 情報科学部 知的制御研究室により作成されたものです。
**スピーカーは別途外付けする必要があります**

音声ファイルの作成には「VOICEVOX:春日部つむぎ」を使用しています。

以下のページのプログラムを参考にしています  
https://github.com/AkariGroup/akari_yolo_inference  
https://github.com/geaxgx/depthai_blazepose.git  
