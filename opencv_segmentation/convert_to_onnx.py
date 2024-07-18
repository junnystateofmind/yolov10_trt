import torch
import torchvision.models.segmentation as models

def convert_to_onnx(model, onnx_path='deeplabv3_mobilenet_v3_large.onnx'):
    dummy_input = torch.randn(1, 3, 520, 520)  # 모델 입력 크기와 맞춤
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11,
                      input_names=['input'], output_names=['output'])
    print(f"Model has been converted to ONNX and saved at {onnx_path}")

if __name__ == "__main__":
    model = models.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.eval()
    convert_to_onnx(model)
