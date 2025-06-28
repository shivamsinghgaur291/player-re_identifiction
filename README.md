# player-re_identifiction
This project implements a player re-identification pipeline using a YOLOv11 model and color histogram-based visual features. The goal is to assign consistent IDs to players across frames of a sports video.


1. Clone or Download the Repository
    git clone https://github.com/shivamsinghgaur291/player-re_identifiction.git
    Change directory to project located

2. Install Python Dependencies  pip install -r requirements.txt

3. Ensure Required Files Are Present
   tracker.py
   utils.py
   best.pt (YOLOv11 weights)
   15sec_input_720p.mp4 (input video)


5. Run the Tracking Script  python tracker.py

A window will pop up showing player detection.

Press Q to quit.

Annotated frames will be saved in tracking_output/

A CSV log will be saved in tracking_logs/player_tracking_log.csv

5. Check the Output
Output images → tracking_output/frame_000.png, frame_001.png, etc.
Logs → tracking_logs/player_tracking_log.csv


