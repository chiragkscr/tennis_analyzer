import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc= torch.nn.Linear(self.model.fc.in_features,14*2) #14 keypoints x and y
        self.model.load_state_dict(torch.load(model_path))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=.5, std=.5)
        ])

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor= self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()

        og_h, og_w= img_rgb.shape[:2]
 
        keypoints[::2] *=  og_w/224.0
        keypoints[1::2] *= og_h/224.0

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            cv2.putText(image, str(i//2), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            cv2.circle(image, (x,y),5,(0,0,255), -1)
        
        return image
    
    def draw_keypoints_on_video(self, video_frames, key_points):
        output_video_frames=[]
        for frame in video_frames:
            frame=self.draw_keypoints(frame, key_points)
            output_video_frames.append(frame)
        
        return output_video_frames