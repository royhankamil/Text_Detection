import json
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import torchvision
    from torchvision import transforms
except Exception as e:
    print("torchvision is required for this script. Please install with: pip install torchvision")
    raise


def build_resnet18_head(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Construct a ResNet18 backbone and replace the final fully-connected layer."""
    from torchvision.models import resnet18
    try:
        # Newer API prefers weights argument
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    except Exception:
        # Fallback to older API
        model = resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_labels(labels_path: Path) -> dict:
    """Load the label mapping from JSON file."""
    with open(labels_path, 'r') as f:
        data = json.load(f)
    # Keys may be strings; convert to int
    return {int(k): v for k, v in data.items()}


def load_model_and_transforms(model_path: Path, labels_path: Path, device: torch.device):
    """Load the trained model and prepare transforms."""
    idx_to_label = load_labels(labels_path)
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get('config', {})
    img_size = int(cfg.get('img_size', 128))
    channels = int(cfg.get('channels', 3))
    mean = cfg.get('mean', [0.5] * channels)
    std = cfg.get('std', [0.5] * channels)

    model = build_resnet18_head(num_classes=len(idx_to_label), pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tfms, idx_to_label, img_size, channels


def preprocess_image_for_model(img: np.ndarray, tfms, device: torch.device) -> torch.Tensor:
    """Convert numpy image to tensor for model inference."""
    # Convert BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    x = tfms(pil_img).unsqueeze(0).to(device)
    return x


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    """Apply softmax to logits to get probabilities."""
    return torch.softmax(logits.float(), dim=1)


def predict_letter(model, image: np.ndarray, tfms, device: torch.device, label_list: List[str], top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Predict letter and return top-k predictions."""
    x = preprocess_image_for_model(image, tfms, device)
    
    with torch.no_grad():
        logits = model(x)
        probs = softmax_logits(logits).cpu().squeeze(0)
        
        # Get top-k predictions
        values, indices = torch.topk(probs, top_k)
        
        pred_label = label_list[int(indices[0])]
        confidence = float(values[0])
        
        top_predictions = [(label_list[int(indices[i])], float(values[i])) for i in range(top_k)]
        
        return pred_label, confidence, top_predictions


def predict_letter_from_file(image_path: str, model_path: str = None, labels_path: str = None, show_image: bool = False):
    """Predict letter from an image file path."""
    
    # Set default paths if not provided
    if model_path is None:
        model_path = str(Path('result') / 'handwriting_resnet18_best.pt')
    if labels_path is None:
        labels_path = str(Path('result') / 'labels.json')
    
    # Check files exist
    input_path = Path(image_path)
    model_path = Path(model_path)
    labels_path = Path(labels_path)
    
    if not input_path.exists():
        print(f"Input image not found: {input_path}")
        return None
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return None
    
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return None
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and labels
    print("Loading model and labels...")
    model, tfms, idx_to_label, img_size, channels = load_model_and_transforms(model_path, labels_path, device)
    label_list = [idx_to_label[i] for i in range(len(idx_to_label))]
    print(f"Loaded model with {len(label_list)} classes")
    
    # Load image
    print(f"Loading image: {input_path}")
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Could not load image: {input_path}")
        return None
    print(f"Image shape: {image.shape}")
    
    # Predict letter
    print("Predicting letter...")
    pred_label, confidence, top_predictions = predict_letter(
        model, image, tfms, device, label_list, 5
    )
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Input image: {input_path}")
    print(f"Predicted letter: {pred_label}")
    print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    print("\nTop predictions:")
    for i, (label, conf) in enumerate(top_predictions, 1):
        print(f"  {i}. {label} - {conf:.3f} ({conf*100:.1f}%)")
    print("="*50)
    
    # Show image if requested
    if show_image:
        # Resize image for display if too large
        display_img = image.copy()
        h, w = display_img.shape[:2]
        if h > 400 or w > 400:
            scale = min(400/h, 400/w)
            new_h, new_w = int(h * scale), int(w * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        # Add prediction text to image
        text = f"Predicted: {pred_label} ({confidence*100:.1f}%)"
        cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Letter Prediction', display_img)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return pred_label, confidence, top_predictions


def main():
    """Interactive main function."""
    print("Simple Letter Predictor")
    print("="*30)
    
    # Get image path from user
    image_path = input("Enter the path to your letter image: ").strip()
    
    if not image_path:
        print("No image path provided. Exiting.")
        return
    
    # Ask if user wants to show the image
    show_img = input("Show the image with prediction? (y/n): ").strip().lower() == 'y'
    
    # Make prediction
    result = predict_letter_from_file(image_path, show_image=show_img)
    
    if result:
        pred_label, confidence, top_predictions = result
        print(f"\nFinal result: The image contains the letter '{pred_label}' with {confidence*100:.1f}% confidence!")


if __name__ == '__main__':
    main()
