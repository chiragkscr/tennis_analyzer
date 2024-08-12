from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append("../")
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model= YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detection=[]

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detection = pickle.load(f)
            return player_detection

        
        for frame in frames:
            player_dict=self.detect_frame(frame)
            player_detection.append(player_dict)
        
        if stub_path is not None:
             with open(stub_path, "wb") as f:
                 pickle.dump(player_detection, f)

        return player_detection

    def detect_frame(self, frame):
        results= self.model.track(frame, persist=True)[0]
        id_name_dict= results.names


        player_dict={}
        for box in results.boxes:
            track_id= int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id= box.cls.tolist()[0]
            object_cls_name=id_name_dict[object_cls_id]
            if object_cls_name=='person':
                player_dict[track_id] = result
            
        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        out_video_frames=[]
        for frame, playerdict in zip(video_frames, player_detections): #zip allows to loop over two list
            #draw boudning boxes
            for track_id, bbox in playerdict.items():
                x1,y1,x2,y2=bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1]) -10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
                frame= cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,255),2)
            out_video_frames.append(frame)

        return out_video_frames
    

    def choose_and_filters(self, court_keypoints, player_detections):
        #Filter players based on Approximity 
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filter_player_detections= []
        for playerdict in player_detections:
            filtered_player_dict= {track_id: bbox for track_id, bbox in playerdict.items() if track_id in chosen_player}
            filter_player_detections.append(filtered_player_dict)

        return filter_player_detections
    
    
    
    
    def choose_players(self, court_keypoints, playerdict):
        distances=[]
        for track_id, bbox in playerdict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
                
            distances.append((track_id, min_distance))
        
        #sort the distances in ascending order 
        distances.sort(key=lambda x: x[1])
        #choose the first 2 tracks
        chosen_players= [distances[0][0], distances[1][0]]
        return chosen_players


