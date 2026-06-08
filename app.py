import gradio as gr
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

device = torch.device("cpu")

model = smp.FPN(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=8
)

model.load_state_dict(torch.load("best_model_fpn.pth", map_location=device))

model.to(device)
model.eval()

COLOR_MAP = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [255, 255, 0],
    [0, 0, 255],
    [128, 128, 128],
    [0, 128, 0],
    [0, 255, 0],
    [255, 255, 255]
], dtype=np.uint8)

def process_image(input_img):
    if input_img is None:
        return None
        
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    h, w, _ = input_img.shape
    transformed = transform(image=input_img)
    tensor_img = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_img)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    color_mask = COLOR_MAP[mask]
    color_mask = cv2.resize(color_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return color_mask

with gr.Blocks() as demo:
    gr.Markdown("### Модуль семантической сегментации спутниковых снимков")
    gr.Markdown("Тестирование архитектуры FPN на датасете LoveDA (класс Urban).")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Входное изображение")
            submit_btn = gr.Button("Запустить инференс", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="Результат сегментации")
            
    submit_btn.click(fn=process_image, inputs=image_input, outputs=image_output)

demo.launch()