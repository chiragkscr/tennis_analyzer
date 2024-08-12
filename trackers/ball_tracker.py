from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model= YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[])for x in ball_positions]
        #convert the list into pandas dataframe
        df_ball_position=pd.DataFrame(ball_positions, columns=['x1', 'y1','x2','y2' ])

        #interpolate the missing values
        df_ball_position = df_ball_position.interpolate()
        df_ball_position = df_ball_position.bfill()

        #convert back to list
        ball_positions = [{1:x} for x in df_ball_position.to_numpy().tolist()]

        return ball_positions
    
    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1,[])for x in ball_positions]
        #convert the list into pandas dataframe
        df_ball_positions=pd.DataFrame(ball_positions, columns=['x1', 'y1','x2','y2' ])

        df_ball_positions["mid_y"] = (df_ball_positions['y1']+ df_ball_positions["y2"])/2

        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()

        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        df_ball_positions['ball_hit']= 0
        mininmum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions)-int(mininmum_change_frames_for_hit*1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1]<0
            positive_poistion_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1]>0

            if negative_position_change or positive_poistion_change:
                change_count = 0
                for change_frame in range(i+1, i+int(mininmum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame]<0
                    positive_poistion_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame]>0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count +=1
                    elif positive_poistion_change and positive_poistion_change_following_frame:
                        change_count +=1
                if change_count>mininmum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i]=1 
        frame_nums = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
        return frame_nums

    def ball_detect_frames(self, frames, read_from_stub=False, stub_path=None):
        Ball_detection=[]

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                Ball_detection = pickle.load(f)
            return Ball_detection

        
        for frame in frames:
            ball_dict=self.detect_frame(frame)
            Ball_detection.append(ball_dict)
        
        if stub_path is not None:
             with open(stub_path, "wb") as f:
                 pickle.dump(Ball_detection, f)

        return Ball_detection

    def detect_frame(self, frame):
        results= self.model.predict(frame, conf=.15)[0]
        


        Ball_dict={}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            Ball_dict[1] = result
            
        return Ball_dict
    
    def draw_bboxes(self, video_frames, player_detection):
        out_video_frames=[]
        for frame, balldict in zip(video_frames, player_detection): #zip allows to loop over two list
            #draw boudning boxes
            for track_id, bbox in balldict.items():
                x1,y1,x2,y2=bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1]) -10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
                frame= cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0),2)
            out_video_frames.append(frame)

        return out_video_frames

