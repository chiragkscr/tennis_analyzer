from utils import (read_video, 
                   save_video, measure_distance, draw_player_stats, convert_pixel_distance_meters)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt
import pandas as pd
from copy import deepcopy
import constants

def main():
    #Read Video
    input_video_path="input_video.mp4"
    video_frames=read_video(input_video_path)

    

    #Detect Players

    player_tracker= PlayerTracker('yolov8x')
    ball_tracker = BallTracker(model_path= "models/best(2).pt")
    player_detection= player_tracker.detect_frames(video_frames, 

                                                   read_from_stub=True, 
                                                   stub_path="tracker_stubs/player_detections.pk1")

    ball_detection= ball_tracker.ball_detect_frames(video_frames,
                                              read_from_stub=True,
                                              stub_path="tracker_stubs/ball_detections.pk1")
    
    

    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)
    
    #Court Line detector model 
    court_model_path = "models/keypoints_model(1).pth"
    court_line_detector= CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    print(court_keypoints)


    #chose playeres
    player_detection = player_tracker.choose_and_filters(court_keypoints, player_detection)
    
    #minicourt 
    mini_court=MiniCourt(video_frames[0])
    
    
    #DRAW output

    ##Draw player bounding boxes
    output_video_frames=player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames=ball_tracker.draw_bboxes(output_video_frames, ball_detection)

    


    ##Draw court keypoints
    output_video_frames= court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)


    


    #detect ball shots
    ball_shots_frames = ball_tracker.get_ball_shot_frames(ball_detection)
    # print(type(ball_shots_frames))

    #convert positons to mini court positions
    player_mini_court_detection, ball_mini_court_detection = mini_court.convert_bounding_boxes2mini_court_coord(player_detection,
                                                                                                                    ball_detection,
                                                                                                                   court_keypoints)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]

    for ball_shot_ind in range(len(ball_shots_frames)-1):
        start_frame = ball_shots_frames[ball_shot_ind]
        end_frame = ball_shots_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detection[start_frame][1],
                                                           ball_mini_court_detection[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detection[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detection[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detection[start_frame][opponent_player_id],
                                                                player_mini_court_detection[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']


    # print(ball_shots_frames)
    #draw mini_court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_minicourt(output_video_frames, player_mini_court_detection)
    output_video_frames = mini_court.draw_points_on_minicourt(output_video_frames, ball_mini_court_detection, color=(0,255,255))


    #draw player stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)
    # draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)

    save_video(output_video_frames, "output_video.avi")


    


if __name__=="__main__":
    main()

