import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional
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

try:
	import tkinter as tk
	from tkinter import filedialog
	FILE_DIALOG_AVAILABLE = True
except ImportError:
	FILE_DIALOG_AVAILABLE = False
	print("tkinter not available for file dialog. Please specify input image path.")


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
	with open(labels_path, 'r') as f:
		data = json.load(f)
	# Keys may be strings; convert to int
	return {int(k): v for k, v in data.items()}


def load_model_and_transforms(model_path: Path, labels_path: Path, device: torch.device):
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


def detect_characters(image: np.ndarray, min_area: int = 100, max_area: int = 10000) -> List[Tuple[int, int, int, int]]:
	"""
	Detect potential character regions in the image.
	Returns list of (x, y, w, h) bounding boxes.
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Apply threshold to get binary image
	_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	# Find contours
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# Filter contours by area and aspect ratio
	character_boxes = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if min_area <= area <= max_area:
			x, y, w, h = cv2.boundingRect(contour)
			# Filter by aspect ratio (characters are usually roughly square-ish)
			aspect_ratio = w / h
			if 0.2 <= aspect_ratio <= 3.0:
				character_boxes.append((x, y, w, h))
	
	# Sort by x-coordinate (left to right)
	character_boxes.sort(key=lambda box: box[0])
	
	return character_boxes


def extract_character_region(image: np.ndarray, box: Tuple[int, int, int, int], padding: int = 10) -> np.ndarray:
	"""Extract a character region with padding."""
	x, y, w, h = box
	h_img, w_img = image.shape[:2]
	
	# Add padding
	x1 = max(0, x - padding)
	y1 = max(0, y - padding)
	x2 = min(w_img, x + w + padding)
	y2 = min(h_img, y + h + padding)
	
	return image[y1:y2, x1:x2]


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
	return torch.softmax(logits.float(), dim=1)


def predict_character(model, char_img: np.ndarray, tfms, device: torch.device, label_list: List[str], top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]]]:
	"""Predict character and return top-k predictions."""
	x = preprocess_image_for_model(char_img, tfms, device)
	
	with torch.no_grad():
		logits = model(x)
		probs = softmax_logits(logits).cpu().squeeze(0)
		
		# Get top-k predictions
		values, indices = torch.topk(probs, top_k)
		
		pred_label = label_list[int(indices[0])]
		confidence = float(values[0])
		
		top_predictions = [(label_list[int(indices[i])], float(values[i])) for i in range(top_k)]
		
		return pred_label, confidence, top_predictions


def draw_results(image: np.ndarray, character_boxes: List[Tuple[int, int, int, int]], 
				predictions: List[Tuple[str, float, List[Tuple[str, float]]]], 
				show_confidence: bool = True, show_top_k: bool = False) -> np.ndarray:
	"""Draw bounding boxes and predictions on the image."""
	result_img = image.copy()
	
	for i, (box, (pred_label, confidence, top_predictions)) in enumerate(zip(character_boxes, predictions)):
		x, y, w, h = box
		
		# Draw bounding box
		cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
		# Draw character number
		cv2.putText(result_img, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		
		# Draw prediction
		text = pred_label
		if show_confidence:
			text += f" ({confidence:.2f})"
		
		# Position text below the box
		text_y = y + h + 20
		cv2.putText(result_img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.putText(result_img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
		
		# Show top-k predictions if requested
		if show_top_k and len(top_predictions) > 1:
			for j, (label, conf) in enumerate(top_predictions[1:], 1):
				alt_text = f"  {j}. {label} ({conf:.2f})"
				alt_y = text_y + 15 * j
				cv2.putText(result_img, alt_text, (x, alt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
	
	return result_img


def select_image_file() -> Optional[Path]:
	"""Open file dialog to select an image file."""
	if not FILE_DIALOG_AVAILABLE:
		return None
	
	# Create and hide the main window
	root = tk.Tk()
	root.withdraw()
	
	# Open file dialog
	file_path = filedialog.askopenfilename(
		title="Select image file to process",
		filetypes=[
			("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
			("All files", "*.*")
		]
	)
	
	root.destroy()
	
	if file_path:
		return Path(file_path)
	return None


def main():
	parser = argparse.ArgumentParser(description="Process an image file to detect and predict handwritten characters.")
	parser.add_argument('input_image', type=str, nargs='?', help='Path to input image file (optional - will open file dialog if not specified)')
	parser.add_argument('--model', type=str, default=str(Path('result') / 'handwriting_resnet18_best.pt'), help='Path to model checkpoint (.pt)')
	parser.add_argument('--labels', type=str, default=str(Path('result') / 'labels.json'), help='Path to labels JSON')
	parser.add_argument('--output', type=str, help='Path to save output image (optional)')
	parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device selection')
	parser.add_argument('--min-area', type=int, default=100, help='Minimum contour area for character detection')
	parser.add_argument('--max-area', type=int, default=10000, help='Maximum contour area for character detection')
	parser.add_argument('--padding', type=int, default=10, help='Padding around detected characters')
	parser.add_argument('--top-k', type=int, default=3, help='Show top-K predictions for each character')
	parser.add_argument('--no-confidence', action='store_true', help='Hide confidence scores')
	parser.add_argument('--show-alternatives', action='store_true', help='Show alternative predictions')
	parser.add_argument('--save-individuals', type=str, help='Directory to save individual character images')
	args = parser.parse_args()

	# Get input image path
	if args.input_image:
		input_path = Path(args.input_image)
	else:
		print("No input image specified. Opening file dialog...")
		input_path = select_image_file()
		if input_path is None:
			print("No file selected. Exiting.")
			sys.exit(1)

	# Check input file
	if not input_path.exists():
		print(f"Input image not found: {input_path}")
		sys.exit(1)

	# Check model and labels
	model_path = Path(args.model)
	labels_path = Path(args.labels)
	if not model_path.exists():
		print(f"Model file not found: {model_path}")
		sys.exit(1)
	if not labels_path.exists():
		print(f"Labels file not found: {labels_path}")
		sys.exit(1)

	# Device selection
	if args.device == 'auto':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device(args.device)
	print(f"Using device: {device}")

	# Load model and labels
	model, tfms, idx_to_label, img_size, channels = load_model_and_transforms(model_path, labels_path, device)
	label_list = [idx_to_label[i] for i in range(len(idx_to_label))]
	print(f"Loaded model with {len(label_list)} classes")

	# Load image
	image = cv2.imread(str(input_path))
	if image is None:
		print(f"Could not load image: {input_path}")
		sys.exit(1)
	print(f"Loaded image: {image.shape}")

	# Detect characters
	print("Detecting characters...")
	character_boxes = detect_characters(image, args.min_area, args.max_area)
	print(f"Found {len(character_boxes)} potential characters")

	if len(character_boxes) == 0:
		print("No characters detected. Try adjusting --min-area and --max-area parameters.")
		sys.exit(1)

	# Create output directory for individual characters if requested
	if args.save_individuals:
		output_dir = Path(args.save_individuals)
		output_dir.mkdir(parents=True, exist_ok=True)

	# Process each character
	predictions = []
	for i, box in enumerate(character_boxes):
		# Extract character region
		char_img = extract_character_region(image, box, args.padding)
		
		# Save individual character if requested
		if args.save_individuals:
			char_path = output_dir / f"char_{i+1:03d}.png"
			cv2.imwrite(str(char_path), char_img)
		
		# Predict character
		pred_label, confidence, top_predictions = predict_character(
			model, char_img, tfms, device, label_list, args.top_k
		)
		predictions.append((pred_label, confidence, top_predictions))
		
		print(f"Character {i+1}: {pred_label} (confidence: {confidence:.3f})")

	# Draw results
	result_image = draw_results(
		image, character_boxes, predictions, 
		show_confidence=not args.no_confidence,
		show_top_k=args.show_alternatives
	)

	# Save output image
	if args.output:
		output_path = Path(args.output)
		cv2.imwrite(str(output_path), result_image)
		print(f"Saved result to: {output_path}")

	# Display results
	assembled_text = ''.join([pred[0] for pred in predictions])
	print(f"\nAssembled text: {assembled_text}")
	
	# Show image
	cv2.imshow('Character Detection Results', result_image)
	print("Press any key to close...")
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main() 