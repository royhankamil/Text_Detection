import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Deque

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


def preprocess_frame(frame_bgr: np.ndarray, roi_ratio: float, roi_size_px: Optional[int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
	"""Center-crop an ROI from the frame. Returns (roi_bgr, (x1,y1,x2,y2))."""
	h, w = frame_bgr.shape[:2]
	if roi_size_px is not None and roi_size_px > 0:
		side = min(roi_size_px, h, w)
	else:
		side = int(min(h, w) * float(roi_ratio))
		side = max(32, min(side, min(h, w)))
	x1 = (w - side) // 2
	y1 = (h - side) // 2
	x2 = x1 + side
	y2 = y1 + side
	roi = frame_bgr[y1:y2, x1:x2]
	return roi, (x1, y1, x2, y2)


def to_tensor_from_bgr(roi_bgr: np.ndarray, tfms, device: torch.device) -> torch.Tensor:
	# Convert BGR -> RGB -> PIL Image
	rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
	pil_img = Image.fromarray(rgb)
	x = tfms(pil_img).unsqueeze(0).to(device)
	return x


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
	return torch.softmax(logits.float(), dim=1)


def render_multiline_text(img, text: str, org: Tuple[int, int], color=(255, 255, 255), scale=0.8, thickness=2, line_gap=8, max_width_px: int = 800):
	"""Render text with basic wrapping."""
	words = text.split(' ')
	lines = []
	cur = ''
	for w in words:
		candidate = (cur + ' ' + w).strip()
		size = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
		if size[0] > max_width_px and cur:
			lines.append(cur)
			cur = w
		else:
			cur = candidate
	if cur:
		lines.append(cur)
	(x, y) = org
	for i, line in enumerate(lines):
		# shadow
		cv2.putText(img, line, (x, y + i * (int(30 * scale) + line_gap)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
		cv2.putText(img, line, (x, y + i * (int(30 * scale) + line_gap)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
	parser = argparse.ArgumentParser(description="Real-time handwriting prediction from webcam using trained model in result/.")
	parser.add_argument('--model', type=str, default=str(Path('result') / 'handwriting_resnet18_best.pt'), help='Path to model checkpoint (.pt)')
	parser.add_argument('--labels', type=str, default=str(Path('result') / 'labels.json'), help='Path to labels JSON')
	parser.add_argument('--camera-index', type=int, default=0, help='OpenCV camera index')
	parser.add_argument('--width', type=int, default=1280, help='Requested camera capture width')
	parser.add_argument('--height', type=int, default=720, help='Requested camera capture height')
	parser.add_argument('--roi-ratio', type=float, default=0.6, help='Center ROI ratio of the min(frame_h, frame_w) [0..1]')
	parser.add_argument('--roi-size', type=int, default=0, help='If > 0, overrides roi-ratio with a fixed square ROI size in pixels')
	parser.add_argument('--no-mirror', action='store_true', help='Disable horizontal flip of the preview')
	parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device selection')
	parser.add_argument('--smoothing', type=int, default=5, help='Temporal smoothing window over probabilities (frames)')
	parser.add_argument('--topk', type=int, default=3, help='Show top-K predictions')
	# New: sentence assembly controls
	parser.add_argument('--assemble', action='store_true', help='Enable sentence assembly from stable character predictions')
	parser.add_argument('--commit-threshold', type=float, default=0.75, help='Confidence threshold to consider committing a character')
	parser.add_argument('--stable-frames', type=int, default=6, help='Number of consecutive stable frames to commit a character')
	parser.add_argument('--gap-frames', type=int, default=20, help='Frames without commit before auto-inserting a space')
	parser.add_argument('--lowconf-threshold', type=float, default=0.40, help='Below this confidence, a frame counts as a gap frame')
	parser.add_argument('--char-candidates', type=int, default=5, help='How many candidate letters to display for the current character')
	parser.add_argument('--max-text-len', type=int, default=256, help='Max assembled text length')
	args = parser.parse_args()

	model_path = Path(args.model)
	labels_path = Path(args.labels)
	if not model_path.exists():
		print(f"Model file not found: {model_path}")
		sys.exit(1)
	if not labels_path.exists():
		print(f"Labels file not found: {labels_path}")
		sys.exit(1)

	if args.device == 'auto':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device(args.device)
	print(f"Using device: {device}")

	model, tfms, idx_to_label, img_size, channels = load_model_and_transforms(model_path, labels_path, device)
	label_list = [idx_to_label[i] for i in range(len(idx_to_label))]

	cap = cv2.VideoCapture(args.camera_index)
	if args.width > 0:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
	if args.height > 0:
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
	if not cap.isOpened():
		print("Could not open camera.")
		sys.exit(1)

	mirror = not args.no_mirror
	smooth_window = max(1, int(args.smoothing))
	prob_queue: Deque[torch.Tensor] = deque(maxlen=smooth_window)

	# State for sentence assembly
	assembled_text = ''
	stable_label_idx: Optional[int] = None
	stable_count = 0
	current_char_probs: list[torch.Tensor] = []
	frames_since_last_commit = 0
	space_inserted = False

	win_name = 'Handwriting Inference (press q to quit)'
	cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

	roi_size_px = args.roi_size if args.roi_size and args.roi_size > 0 else None

	with torch.no_grad():
		while True:
			ret, frame = cap.read()
			if not ret:
				print("Failed to read frame from camera.")
				break

			if mirror:
				frame = cv2.flip(frame, 1)

			roi_bgr, (x1, y1, x2, y2) = preprocess_frame(frame, args.roi_ratio, roi_size_px)
			x = to_tensor_from_bgr(roi_bgr, tfms, device)

			logits = model(x)
			probs = softmax_logits(logits).cpu().squeeze(0)
			prob_queue.append(probs)

			avg_probs = torch.stack(list(prob_queue), dim=0).mean(dim=0) if len(prob_queue) > 0 else probs
			conf, pred_idx = torch.max(avg_probs, dim=0)
			pred_label = label_list[int(pred_idx.item())]
			conf_val = float(conf.item())

			# Sentence assembly logic
			if args.assemble:
				if conf_val >= args.commit_threshold:
					if stable_label_idx is None or int(pred_idx) != int(stable_label_idx):
						stable_label_idx = int(pred_idx)
						stable_count = 1
						current_char_probs = [avg_probs]
					else:
						stable_count += 1
						current_char_probs.append(avg_probs)
					# Commit character when stable long enough
					if stable_count >= args.stable_frames:
						agg = torch.stack(current_char_probs, dim=0).mean(dim=0)
						char_idx = int(torch.argmax(agg).item())
						char_label = label_list[char_idx]
						assembled_text = (assembled_text + char_label)[:args.max_text_len]
						# reset
						stable_label_idx = None
						stable_count = 0
						current_char_probs = []
						frames_since_last_commit = 0
						space_inserted = False
				else:
					frames_since_last_commit += 1
					if conf_val < args.lowconf_threshold and frames_since_last_commit >= args.gap_frames and not space_inserted:
						assembled_text = (assembled_text + ' ')[:args.max_text_len]
						space_inserted = True
			# Overlay info
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
			text = f"Pred: {pred_label}  Conf: {conf_val:.2f}"
			cv2.putText(frame, text, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
			cv2.putText(frame, text, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

			# Top-k display for current frame
			k = max(1, min(args.topk, len(label_list)))
			values, indices = torch.topk(avg_probs, k)
			for i in range(k):
				lbl = label_list[int(indices[i])]
				p = float(values[i])
				display = f"{i+1}. {lbl}: {p:.2f}"
				cv2.putText(frame, display, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
				cv2.putText(frame, display, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

			# Candidate display for current character (aggregated over stable window)
			if args.assemble:
				if current_char_probs:
					agg = torch.stack(current_char_probs, dim=0).mean(dim=0)
				else:
					agg = avg_probs
				cc = max(1, min(args.char_candidates, len(label_list)))
				val2, idx2 = torch.topk(agg, cc)
				cand_text = 'Candidates: ' + '  '.join([f"{label_list[int(idx2[j])]}({float(val2[j]):.2f})" for j in range(cc)])
				cv2.putText(frame, cand_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
				cv2.putText(frame, cand_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

			# Render assembled text
			if args.assemble and assembled_text:
				render_multiline_text(frame, assembled_text, (10, frame.shape[0] - 60), color=(255, 255, 255), scale=0.9, thickness=2, line_gap=6, max_width_px=int(frame.shape[1]*0.9))

			cv2.imshow(win_name, frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			elif key == ord('m'):
				mirror = not mirror
			elif key == ord('['):
				# shrink ROI
				if roi_size_px is None:
					roi_size_px = int(min(frame.shape[:2]) * args.roi_ratio)
				roi_size_px = max(32, int(roi_size_px * 0.9))
			elif key == ord(']'):
				# enlarge ROI
				if roi_size_px is None:
					roi_size_px = int(min(frame.shape[:2]) * args.roi_ratio)
				roi_size_px = min(int(roi_size_px * 1.1), min(frame.shape[0], frame.shape[1]))
			elif args.assemble and key == ord('c'):
				assembled_text = ''
				stable_label_idx = None
				stable_count = 0
				current_char_probs = []
				frames_since_last_commit = 0
				space_inserted = False
			elif args.assemble and key == ord('b'):
				assembled_text = assembled_text[:-1]
			elif args.assemble and key == ord(' '):
				assembled_text = (assembled_text + ' ')[:args.max_text_len]
				space_inserted = True

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main() 