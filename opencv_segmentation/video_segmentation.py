import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import os

def load_deeplab_model(weights_path='deeplabv3_mobilenet_v3_large.pth'):
    if os.path.exists(weights_path):
        print(f"Loading model weights from {weights_path}")
        model = deeplabv3_mobilenet_v3_large(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    else:
        print("Downloading pretrained weights...")
        model = deeplabv3_mobilenet_v3_large(pretrained=True)
        torch.save(model.state_dict(), weights_path)
    model.eval()
    return model

def preprocess_image(image):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

import cv2
import numpy as np
from argparse import ArgumentParser

def mean_brightness(img):
    fixed = 100
    m = cv2.mean(img)
    scalar = (-int(m[0]) + fixed, -int(m[1]) + fixed, -int(m[2]) + fixed, 0)
    dst = cv2.add(img, scalar)
    return dst

def preprocess_frame(frame, hsv=True, blur=False, brightness=False):
    img = frame
    if brightness:
        img = mean_brightness(img)
    if blur:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    if hsv:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img

def select_color(img, color_range):
    selected = cv2.inRange(img, color_range[0], color_range[1])
    return selected

def detect_contours(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours

def segment_frame(frame, model):
    input_tensor = preprocess_image(frame)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions.byte().cpu().numpy()

def apply_segmentation_mask(frame, mask):
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(frame)
    colored_mask[mask == 15] = [0, 255, 0]  # 예: 클래스 15 (사람)를 초록색으로 표시
    return cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)

def process_video(video_path, model):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video file could not be opened!")
        exit(1)

    color_range = np.array([[81, 124, 0], [128, 182, 255]])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 필터링
        preprocessed = preprocess_frame(frame, blur=True)
        hsv_img = select_color(preprocessed, color_range)
        contours = detect_contours(hsv_img)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # DeepLabV3 세그멘테이션
        mask = segment_frame(frame, model)
        segmented_frame = apply_segmentation_mask(frame, mask)

        cv2.imshow("Segmented Video", segmented_frame)

        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()

    model = load_deeplab_model()
    process_video(args.video, model)
