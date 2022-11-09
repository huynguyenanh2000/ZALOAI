import cv2 
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss
import argparse
import os
import pandas as pd 
import statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS.pth'))
model.eval()

model.to(device)


tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
def inference(args):
    result = []
    count = 1
    for file in os.listdir(args.files):
        print(f'{count}: {file}')
        count+=1
        link = args.files+file
        confidence = []
        cap = cv2.VideoCapture(link)
        i = 0
        frame_skip = 10
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            if i > frame_skip - 1:
                frame_count += 1
                faceRegion = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faceRegion = tfms(faceRegion)
                faceRegion = faceRegion.unsqueeze(0)
                mask, binary = model.forward(faceRegion.to(device))
                res = torch.mean(mask).item()
                confidence.append(res)
                i = 0
                continue
            i += 1
        result.append([file,statistics.median(confidence)])
    df  = pd.DataFrame(result, columns = ['fname', 'liveness_score'])
    df.to_csv('./Predict.csv', index = False)

    return res 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, default='video')

    args = parser.parse_args()
    inference(args)