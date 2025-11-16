#!/usr/bin/env python3
"""
Intelligent Instance Pre-Filtering for Character Clustering

PURPOSE: Filter out 80-90% of background objects BEFORE clustering
         to dramatically reduce processing time and improve accuracy.

MULTI-STAGE FILTERING:
1. Geometric Filter: Size, aspect ratio, position
2. Face Detection Filter: Only keep instances with faces
3. Semantic Filter: CLIP-based "character" vs "background" classification
4. Quality Filter: Blur, occlusion, truncation checks

USAGE:
    python instance_prefilter.py \
      --input-dir /path/to/sam2_instances \
      --output-dir /path/to/filtered_instances \
      --mode aggressive \
      --min-face-confidence 0.7
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime


# ============================================================================
# Geometric Filtering
# ============================================================================

class GeometricFilter:
    """Filter based on size, aspect ratio, position"""
    
    def __init__(
        self,
        min_size: int = 128,
        max_size: int = 1536,
        min_aspect: float = 0.3,
        max_aspect: float = 3.0,
        edge_margin: int = 10
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.edge_margin = edge_margin
    
    def filter(self, image: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Check if instance passes geometric criteria
        
        Returns:
            (passed, reason)
        """
        h, w = image.shape[:2]
        aspect = w / h if h > 0 else 0
        
        # Size check
        if min(h, w) < self.min_size:
            return False, f"too_small ({w}x{h})"
        if max(h, w) > self.max_size:
            return False, f"too_large ({w}x{h})"
        
        # Aspect ratio check (characters are usually vertical)
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return False, f"bad_aspect ({aspect:.2f})"
        
        # Edge position check (instances at very edge are often background)
        if metadata and 'bbox' in metadata:
            x, y, w_bbox, h_bbox = metadata['bbox']
            frame_w, frame_h = metadata.get('frame_size', (1920, 1080))
            
            if x < self.edge_margin or y < self.edge_margin:
                return False, "edge_position"
            if x + w_bbox > frame_w - self.edge_margin:
                return False, "edge_position"
            if y + h_bbox > frame_h - self.edge_margin:
                return False, "edge_position"
        
        return True, "ok"


# ============================================================================
# Face Detection Filter
# ============================================================================

class FaceFilter:
    """Filter based on face detection (most reliable for 3D characters)"""
    
    def __init__(self, device: str = "cuda", min_confidence: float = 0.7):
        self.device = device
        self.min_confidence = min_confidence
        self._init_detector()
    
    def _init_detector(self):
        """Initialize face detector (prefer RetinaFace for 3D)"""
        try:
            from retinaface import RetinaFace
            self.detector_type = "retinaface"
            self.detector = RetinaFace
            print("✓ Using RetinaFace for face detection")
        except ImportError:
            try:
                from facenet_pytorch import MTCNN
                self.detector_type = "mtcnn"
                self.detector = MTCNN(device=self.device)
                print("✓ Using MTCNN for face detection")
            except ImportError:
                # Fallback to OpenCV (less accurate)
                self.detector_type = "opencv"
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                print("⚠️ Using OpenCV Haar Cascades (less accurate)")
    
    def filter(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Check if instance contains a face
        
        Returns:
            (has_face, reason)
        """
        if self.detector_type == "retinaface":
            return self._filter_retinaface(image)
        elif self.detector_type == "mtcnn":
            return self._filter_mtcnn(image)
        else:
            return self._filter_opencv(image)
    
    def _filter_retinaface(self, image: np.ndarray) -> Tuple[bool, str]:
        """RetinaFace detection"""
        try:
            # RetinaFace expects BGR
            faces = self.detector.detect_faces(image)
            
            if not faces:
                return False, "no_face"
            
            # Check confidence
            max_confidence = max(face['score'] for face in faces.values())
            if max_confidence < self.min_confidence:
                return False, f"low_confidence ({max_confidence:.2f})"
            
            return True, f"face_detected ({len(faces)} faces)"
        
        except Exception as e:
            return False, f"detection_error: {e}"
    
    def _filter_mtcnn(self, image: np.ndarray) -> Tuple[bool, str]:
        """MTCNN detection"""
        try:
            # MTCNN expects RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            
            boxes, probs = self.detector.detect(pil_img)
            
            if boxes is None or len(boxes) == 0:
                return False, "no_face"
            
            max_conf = max(probs) if probs is not None else 0
            if max_conf < self.min_confidence:
                return False, f"low_confidence ({max_conf:.2f})"
            
            return True, f"face_detected ({len(boxes)} faces)"
        
        except Exception as e:
            return False, f"detection_error: {e}"
    
    def _filter_opencv(self, image: np.ndarray) -> Tuple[bool, str]:
        """OpenCV Haar Cascade detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return False, "no_face"
            
            return True, f"face_detected ({len(faces)} faces)"
        
        except Exception as e:
            return False, f"detection_error: {e}"


# ============================================================================
# Semantic Filter (CLIP-based)
# ============================================================================

class SemanticFilter:
    """CLIP-based semantic classification: character vs background"""
    
    def __init__(self, device: str = "cuda", threshold: float = 0.6):
        self.device = device
        self.threshold = threshold
        self._init_clip()
    
    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Define text prompts
            self.text_prompts = [
                "a 3D animated character person",
                "a Pixar-style animated human character",
                "a cartoon character with a face",
                "background object or scenery",
                "inanimate object or prop",
                "environment or landscape"
            ]
            
            # Encode text prompts
            text_tokens = clip.tokenize(self.text_prompts).to(self.device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(text_tokens)
                self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            
            print("✓ CLIP semantic filter initialized")
        
        except ImportError:
            print("⚠️ CLIP not installed, semantic filtering disabled")
            self.model = None
    
    def filter(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Classify instance as character or background
        
        Returns:
            (is_character, reason)
        """
        if self.model is None:
            return True, "semantic_filter_disabled"
        
        try:
            # Preprocess image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            
            # Encode image
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = (image_features @ self.text_features.T).squeeze(0)
            probs = similarities.softmax(dim=0).cpu().numpy()
            
            # Character prompts: indices 0, 1, 2
            # Background prompts: indices 3, 4, 5
            char_score = probs[:3].sum()
            bg_score = probs[3:].sum()
            
            if char_score > self.threshold:
                return True, f"character (score={char_score:.2f})"
            else:
                return False, f"background (char={char_score:.2f}, bg={bg_score:.2f})"
        
        except Exception as e:
            return True, f"semantic_error: {e}"


# ============================================================================
# Quality Filter
# ============================================================================

class QualityFilter:
    """Filter based on blur, truncation, occlusion"""
    
    def __init__(
        self,
        min_sharpness: float = 50.0,
        max_truncation: float = 0.2
    ):
        self.min_sharpness = min_sharpness
        self.max_truncation = max_truncation
    
    def filter(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Check image quality
        
        Returns:
            (passed, reason)
        """
        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < self.min_sharpness:
            return False, f"too_blurry ({laplacian_var:.1f})"
        
        # Truncation detection (check if alpha channel is cut off at edges)
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            edge_alpha = np.concatenate([
                alpha[0, :],
                alpha[-1, :],
                alpha[:, 0],
                alpha[:, -1]
            ])
            truncation_ratio = (edge_alpha > 200).sum() / len(edge_alpha)
            
            if truncation_ratio > self.max_truncation:
                return False, f"truncated ({truncation_ratio:.2f})"
        
        return True, "ok"


# ============================================================================
# Main Pre-Filter Pipeline
# ============================================================================

class InstancePreFilter:
    """Combined multi-stage filtering pipeline"""
    
    def __init__(
        self,
        mode: str = "balanced",
        device: str = "cuda",
        enable_semantic: bool = True
    ):
        """
        Initialize filters
        
        Args:
            mode: 'conservative' (keep more) | 'balanced' | 'aggressive' (keep fewer)
            device: cuda or cpu
            enable_semantic: Use CLIP semantic filter (slower but more accurate)
        """
        self.mode = mode
        self.device = device
        
        # Initialize filters based on mode
        if mode == "conservative":
            self.geometric = GeometricFilter(min_size=64, min_aspect=0.2, max_aspect=5.0)
            self.face = FaceFilter(device=device, min_confidence=0.5)
            self.quality = QualityFilter(min_sharpness=30.0, max_truncation=0.3)
            semantic_threshold = 0.5
        elif mode == "aggressive":
            self.geometric = GeometricFilter(min_size=192, min_aspect=0.4, max_aspect=2.5)
            self.face = FaceFilter(device=device, min_confidence=0.8)
            self.quality = QualityFilter(min_sharpness=80.0, max_truncation=0.1)
            semantic_threshold = 0.7
        else:  # balanced
            self.geometric = GeometricFilter(min_size=128, min_aspect=0.3, max_aspect=3.0)
            self.face = FaceFilter(device=device, min_confidence=0.7)
            self.quality = QualityFilter(min_sharpness=50.0, max_truncation=0.2)
            semantic_threshold = 0.6
        
        # Semantic filter (optional, slower)
        self.semantic = SemanticFilter(device=device, threshold=semantic_threshold) if enable_semantic else None
        
        print(f"🔧 Pre-filter initialized in '{mode}' mode")
    
    def filter_instance(self, image: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Apply all filters to an instance
        
        Returns:
            (passed, filter_results)
        """
        results = {}
        
        # Stage 1: Geometric filter (fastest)
        passed, reason = self.geometric.filter(image, metadata)
        results['geometric'] = {'passed': passed, 'reason': reason}
        if not passed:
            results['final_decision'] = 'rejected'
            results['rejection_reason'] = f"geometric_{reason}"
            return False, results
        
        # Stage 2: Quality filter
        passed, reason = self.quality.filter(image)
        results['quality'] = {'passed': passed, 'reason': reason}
        if not passed:
            results['final_decision'] = 'rejected'
            results['rejection_reason'] = f"quality_{reason}"
            return False, results
        
        # Stage 3: Face detection (most important for characters)
        passed, reason = self.face.filter(image)
        results['face'] = {'passed': passed, 'reason': reason}
        if not passed:
            results['final_decision'] = 'rejected'
            results['rejection_reason'] = f"face_{reason}"
            return False, results
        
        # Stage 4: Semantic filter (optional, slower but more accurate)
        if self.semantic:
            passed, reason = self.semantic.filter(image)
            results['semantic'] = {'passed': passed, 'reason': reason}
            if not passed:
                results['final_decision'] = 'rejected'
                results['rejection_reason'] = f"semantic_{reason}"
                return False, results
        
        # All filters passed
        results['final_decision'] = 'accepted'
        return True, results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Instance Pre-Filtering for Character Clustering")
    parser.add_argument("--input-dir", type=str, required=True, help="SAM2 instances directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Filtered output directory")
    parser.add_argument("--mode", type=str, choices=["conservative", "balanced", "aggressive"],
                        default="balanced", help="Filtering aggressiveness")
    parser.add_argument("--enable-semantic", action="store_true", help="Enable CLIP semantic filter (slower)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch-report", type=str, help="Save batch filtering report (JSON)")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = "cpu"
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rejected_dir = output_dir / "rejected"
    rejected_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"INSTANCE PRE-FILTERING - {args.mode.upper()} MODE")
    print(f"{'='*70}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Semantic filter: {'Enabled' if args.enable_semantic else 'Disabled'}")
    print(f"{'='*70}\n")
    
    # Initialize filter
    prefilter = InstancePreFilter(
        mode=args.mode,
        device=args.device,
        enable_semantic=args.enable_semantic
    )
    
    # Get all instance images
    image_files = []
    for ext in ['*.png', '*.jpg']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📂 Found {len(image_files)} instances to filter\n")
    
    # Process instances
    stats = {
        'total': len(image_files),
        'accepted': 0,
        'rejected': 0,
        'rejection_reasons': {}
    }
    
    results_log = []
    
    for img_path in tqdm(image_files, desc="Filtering"):
        try:
            # Load image
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            
            # Apply filters
            passed, filter_results = prefilter.filter_instance(image)
            
            # Save result
            if passed:
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), image)
                stats['accepted'] += 1
            else:
                # Save to rejected folder (for debugging)
                reject_path = rejected_dir / img_path.name
                cv2.imwrite(str(reject_path), image)
                stats['rejected'] += 1
                
                reason = filter_results.get('rejection_reason', 'unknown')
                stats['rejection_reasons'][reason] = stats['rejection_reasons'].get(reason, 0) + 1
            
            # Log results
            results_log.append({
                'filename': img_path.name,
                'passed': passed,
                'filters': filter_results
            })
        
        except Exception as e:
            print(f"\n⚠️ Error processing {img_path.name}: {e}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"FILTERING COMPLETE")
    print(f"{'='*70}")
    print(f"Total instances: {stats['total']}")
    print(f"Accepted (characters): {stats['accepted']} ({stats['accepted']/stats['total']*100:.1f}%)")
    print(f"Rejected (background): {stats['rejected']} ({stats['rejected']/stats['total']*100:.1f}%)")
    print(f"\nRejection reasons:")
    for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"{'='*70}\n")
    
    # Save report
    if args.batch_report:
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': args.mode,
            'semantic_enabled': args.enable_semantic,
            'statistics': stats,
            'per_instance_results': results_log
        }
        
        with open(args.batch_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📝 Detailed report saved: {args.batch_report}")


if __name__ == "__main__":
    main()
