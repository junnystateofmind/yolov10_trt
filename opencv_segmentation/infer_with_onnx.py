import onnxruntime as ort
import cv2
import numpy as np
from argparse import ArgumentParser

def preprocess_image(image):
    # 이미지 전처리
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (520, 520))
    image = image.astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0).astype(np.float32)

def apply_segmentation_mask(frame, mask):
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(frame)
    colored_mask[mask == 15] = [0, 255, 0]  # 예: 클래스 15 (사람)를 초록색으로 표시
    return cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)

def process_video(video_path, onnx_model_path):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video file could not be opened!")
        exit(1)

    # ONNX 런타임 세션 생성
    ort_session = ort.InferenceSession(onnx_model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_image = preprocess_image(frame)
        ort_inputs = {ort_session.get_inputs()[0].name: input_image}
        ort_outs = ort_session.run(None, ort_inputs)
        mask = ort_outs[0].argmax(1).squeeze(0).astype(np.uint8)

        segmented_frame = apply_segmentation_mask(frame, mask)

        cv2.imshow("Segmented Video", segmented_frame)

        if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file")
    args = parser.parse_args()

    process_video(args.video, args.onnx)
