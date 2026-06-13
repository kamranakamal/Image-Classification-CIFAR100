from PIL import Image
import torch
import time
import gradio as gr
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import test_transforms
import torchvision
from src.Helper import load_cifar_classes
from src.effnet_model import create_effnet


model = create_effnet(weights=False)

project_root = Path(__file__).parent.parent
model_path = project_root / "models" / "best_model.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))
class_names = load_cifar_classes(str(Path(__file__).parent / "classes.txt"))

def predict(image):
    # start timer
    start_time = time.time()

    # numpy -> PIL
    image = Image.fromarray(image).convert("RGB")

    # transforms
    image = test_transforms(image)

    # add batch dimension
    image = image.unsqueeze(0).to("cpu")

    model.eval()

    with torch.inference_mode():
        logits = model(image)

        # probabilities
        probs = torch.softmax(logits, dim=1)

        # top 3 predictions
        top_probs, top_indices = torch.topk(probs, k=3)

    # prediction time
    end_time = time.time()
    pred_time = end_time - start_time

    # convert tensors -> python values
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    # create output dictionary
    predictions = {
        class_names[idx]: float(prob)
        for idx, prob in zip(top_indices, top_probs)
    }

    # top class label
    pred_label = class_names[top_indices[0]]

    return (
        predictions,
        pred_label,
        f"{pred_time:.4f} seconds"
    )

    


if __name__=="__main__":
    demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=[
        gr.Label(num_top_classes=3, label="Top Predictions"),
        gr.Text(label="Predicted Class"),
        gr.Text(label="Prediction Time")])
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=False)
