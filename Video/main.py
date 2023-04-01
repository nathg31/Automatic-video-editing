import sys, torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from video_preprocessing.preprocess import preprocess
from video_segmentation.segment import get_change_points, get_shot_importance
from layers.summarizer import PGL_SUM


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook 

def googleNet():
    model = models.googlenet(pretrained=True)
    model.eval()
    model.avgpool.register_forward_hook(get_activation('pool5'))
    lenet = nn.Sequential(*list(model.children())[:-2])

    return lenet

def get_score_model():
    model_path = "models/scoring_models/PGL-SUM/pretrained_models/table3_models/SumMe/split2/epoch-16.pt"
    trained_model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                                    fusion="add", pos_enc="absolute")
    trained_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return trained_model

def get_features(model, video_res, transform):
    frame_features = []
    for frame in tqdm(video_res['video_frames']):
        img = Image.fromarray(frame)
        input = transform(img)

        # Add an extra dimension to the input tensor
        input = input.unsqueeze(0)

        feat = model(input).squeeze().detach().numpy()
        frame_features.append(feat)
    return frame_features

def main():
    # TODO handle multiple video link
    video_paths = sys.argv[1:]
    all_change_points = []
    all_scores = []
    all_frames = []
    all_positions = []
    model = googleNet()

    # Transform the frames
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    scoring_model = get_score_model()

    for video_path in video_paths:
        # Get the video frames
        video_res = preprocess(video_path, 2)
        print(video_res['n_frame'])

        # Get the features
        frame_features = get_features(model, video_res, transform)
        frame_features_tensor = torch.Tensor(np.array(frame_features)).view(-1, 1024)
        # Get the change points
        change_points, _ = get_change_points(frame_features, video_res['n_frame'])

        # Get the frame importance scores
        with torch.no_grad():
            scores, _ = scoring_model(frame_features_tensor)
        all_change_points.append(change_points)
        all_scores.append(scores[0])
        all_frames.append(video_res['n_frame'])
        all_positions.append(video_res['picks'])

    # Get the shot importance scores
    all_shots_imp_scores = get_shot_importance(all_change_points, 
                                                all_scores,
                                                all_frames,
                                                all_positions,
                                                video_paths)
    print(all_shots_imp_scores)

if __name__ == "__main__":
    main()