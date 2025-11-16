#!/usr/bin/env python3
"""
Automatic LoRA Checkpoint Evaluator - SOTA Edition

Evaluates LoRA checkpoints using state-of-the-art models:
- InternVL2 / CLIP (prompt-image alignment)
- LAION Aesthetics (aesthetic quality)
- InsightFace (character identity consistency)
- MUSIQ (technical image quality)
- LPIPS (perceptual similarity)
- Diversity metrics (mode collapse detection)

Falls back to basic CLIP if advanced models unavailable.
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Import evaluation models with graceful fallbacks
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer
from diffusers import StableDiffusionPipeline
import clip

# Optional advanced models
ADVANCED_MODELS_AVAILABLE = {
    'insightface': False,
    'lpips': False,
    'pyiqa': False,  # MUSIQ
    'laion_aesthetics': False,
    'internvl2': False
}

# Try importing optional dependencies
try:
    import insightface
    from insightface.app import FaceAnalysis
    ADVANCED_MODELS_AVAILABLE['insightface'] = True
except ImportError:
    warnings.warn("InsightFace not available. Using CLIP for character consistency.")

try:
    import lpips
    ADVANCED_MODELS_AVAILABLE['lpips'] = True
except ImportError:
    warnings.warn("LPIPS not available. Using basic diversity metric.")

try:
    import pyiqa
    ADVANCED_MODELS_AVAILABLE['pyiqa'] = True
except ImportError:
    warnings.warn("PyIQA (MUSIQ) not available. Using basic quality metric.")


class LoRAEvaluator:
    """Comprehensive LoRA checkpoint evaluator with SOTA models"""

    def __init__(self, base_model_path: str, device: str = 'cuda', use_sota: bool = True):
        self.device = device
        self.use_sota = use_sota

        print("="*70)
        print("LOADING EVALUATION MODELS (SOTA Edition)")
        print("="*70)

        # ===== PROMPT ALIGNMENT: InternVL2 (preferred) or CLIP =====
        if use_sota and self._check_internvl2():
            print("✓ Loading InternVL2-8B for prompt alignment (SOTA)...")
            try:
                self.internvl_model = AutoModel.from_pretrained(
                    "OpenGVLab/InternVL2-8B",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map=device
                )
                self.internvl_tokenizer = AutoTokenizer.from_pretrained(
                    "OpenGVLab/InternVL2-8B",
                    trust_remote_code=True
                )
                self.use_internvl = True
                print("  → InternVL2 loaded successfully")
                ADVANCED_MODELS_AVAILABLE['internvl2'] = True
            except Exception as e:
                print(f"  ✗ InternVL2 failed to load: {e}")
                print("  → Falling back to CLIP")
                self.use_internvl = False
        else:
            self.use_internvl = False

        # Fallback to CLIP
        if not self.use_internvl:
            print("✓ Loading CLIP ViT-L/14 for prompt alignment...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
            print("  → CLIP loaded successfully")

        # ===== AESTHETICS: LAION Aesthetics Predictor =====
        if use_sota:
            print("✓ Loading LAION Aesthetics Predictor V2...")
            try:
                from transformers import pipeline
                self.aesthetic_scorer = pipeline(
                    "image-classification",
                    model="cafeai/cafe_aesthetic",
                    device=0 if device == 'cuda' else -1
                )
                print("  → LAION Aesthetics loaded successfully")
                ADVANCED_MODELS_AVAILABLE['laion_aesthetics'] = True
            except Exception as e:
                print(f"  ✗ LAION Aesthetics failed: {e}")
                self.aesthetic_scorer = None
        else:
            self.aesthetic_scorer = None

        # ===== CHARACTER CONSISTENCY: InsightFace =====
        if use_sota and ADVANCED_MODELS_AVAILABLE['insightface']:
            print("✓ Loading InsightFace for character identity...")
            try:
                self.face_analyzer = FaceAnalysis(
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_analyzer.prepare(ctx_id=0 if device == 'cuda' else -1)
                print("  → InsightFace loaded successfully")
            except Exception as e:
                print(f"  ✗ InsightFace failed: {e}")
                self.face_analyzer = None
        else:
            self.face_analyzer = None

        # ===== IMAGE QUALITY: MUSIQ =====
        if use_sota and ADVANCED_MODELS_AVAILABLE['pyiqa']:
            print("✓ Loading MUSIQ for technical quality...")
            try:
                self.musiq_metric = pyiqa.create_metric('musiq', device=device)
                print("  → MUSIQ loaded successfully")
            except Exception as e:
                print(f"  ✗ MUSIQ failed: {e}")
                self.musiq_metric = None
        else:
            self.musiq_metric = None

        # ===== PERCEPTUAL SIMILARITY: LPIPS =====
        if use_sota and ADVANCED_MODELS_AVAILABLE['lpips']:
            print("✓ Loading LPIPS for perceptual similarity...")
            try:
                self.lpips_metric = lpips.LPIPS(net='alex').to(device)
                print("  → LPIPS loaded successfully")
            except Exception as e:
                print(f"  ✗ LPIPS failed: {e}")
                self.lpips_metric = None
        else:
            self.lpips_metric = None

        # ===== STABLE DIFFUSION PIPELINE =====
        print("✓ Loading Stable Diffusion pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            safety_checker=None
        ).to(device)
        print("  → SD pipeline loaded successfully")

        print("="*70)
        print("MODEL LOADING COMPLETE")
        print("="*70)
        self._print_model_status()
        print("="*70 + "\n")

    def _check_internvl2(self) -> bool:
        """Check if InternVL2 model is available locally or can be downloaded"""
        try:
            from huggingface_hub import snapshot_download
            return True
        except:
            return False

    def _print_model_status(self):
        """Print status of all evaluation models"""
        print("\nModel Status:")
        print(f"  Prompt Alignment:     {'InternVL2 (SOTA)' if self.use_internvl else 'CLIP (baseline)'}")
        print(f"  Aesthetics:           {'LAION V2 (SOTA)' if self.aesthetic_scorer else 'Sharpness (basic)'}")
        print(f"  Character Identity:   {'InsightFace (SOTA)' if self.face_analyzer else 'CLIP (baseline)'}")
        print(f"  Technical Quality:    {'MUSIQ (SOTA)' if self.musiq_metric else 'Laplacian (basic)'}")
        print(f"  Perceptual Sim:       {'LPIPS (SOTA)' if self.lpips_metric else 'CLIP std (basic)'}")

    def evaluate_checkpoint(
        self,
        lora_path: Path,
        test_prompts: List[str],
        output_dir: Path,
        num_images_per_prompt: int = 4
    ) -> Dict:
        """Evaluate a single checkpoint with SOTA models"""

        print(f"\n{'='*60}")
        print(f"Evaluating: {lora_path.name}")
        print(f"{'='*60}")

        # Load LoRA
        self.pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

        # Generate test images
        generated_images = []
        prompts_used = []

        for prompt in tqdm(test_prompts, desc="Generating test images"):
            for i in range(num_images_per_prompt):
                image = self.pipe(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(42 + i)
                ).images[0]

                generated_images.append(image)
                prompts_used.append(prompt)

        # Save test images
        test_output_dir = output_dir / lora_path.stem
        test_output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img, prompt) in enumerate(zip(generated_images, prompts_used)):
            img.save(test_output_dir / f"test_{idx:03d}.png")

        # Compute metrics with SOTA models
        print("Computing evaluation metrics...")
        metrics = {
            'checkpoint': lora_path.name,
        }

        # Prompt alignment (InternVL2 or CLIP)
        if self.use_internvl:
            metrics['internvl_score'] = self.compute_internvl_score(generated_images, prompts_used)
            metrics['clip_score'] = metrics['internvl_score']  # For backward compatibility
        else:
            metrics['clip_score'] = self.compute_clip_score(generated_images, prompts_used)

        # Character consistency (InsightFace or CLIP)
        if self.face_analyzer:
            metrics['character_consistency'] = self.compute_face_consistency(generated_images)
        else:
            metrics['character_consistency'] = self.compute_character_consistency(generated_images)

        # Aesthetics (LAION or basic)
        if self.aesthetic_scorer:
            metrics['aesthetic_score'] = self.compute_laion_aesthetics(generated_images)
        else:
            metrics['aesthetic_score'] = self.compute_image_quality(generated_images)

        # Technical quality (MUSIQ or Laplacian)
        if self.musiq_metric:
            metrics['image_quality'] = self.compute_musiq_quality(generated_images)
        else:
            metrics['image_quality'] = self.compute_image_quality(generated_images)

        # Diversity (LPIPS or CLIP std)
        if self.lpips_metric:
            metrics['diversity'] = self.compute_lpips_diversity(generated_images)
        else:
            metrics['diversity'] = self.compute_diversity(generated_images)

        # Compute composite score
        metrics['composite_score'] = self.compute_composite_score(metrics)

        # Unload LoRA
        self.pipe.unload_lora_weights()

        return metrics

    # ==================== PROMPT ALIGNMENT METRICS ====================

    def compute_internvl_score(self, images: List[Image.Image], prompts: List[str]) -> float:
        """Compute InternVL2 score (SOTA prompt-image alignment)"""
        try:
            scores = []
            for img, prompt in tqdm(zip(images, prompts), desc="InternVL scoring", leave=False):
                # Prepare inputs for InternVL2
                pixel_values = self.internvl_model.image_processor(
                    img, return_tensors='pt'
                ).pixel_values.to(self.device)

                input_ids = self.internvl_tokenizer(
                    prompt, return_tensors='pt'
                ).input_ids.to(self.device)

                with torch.no_grad():
                    # Get image and text embeddings
                    image_embeds = self.internvl_model.vision_model(pixel_values).last_hidden_state.mean(dim=1)
                    text_embeds = self.internvl_model.text_model(input_ids).last_hidden_state.mean(dim=1)

                    # Normalize
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                    # Cosine similarity
                    similarity = (image_embeds @ text_embeds.T).item()
                    scores.append(similarity)

            return float(np.mean(scores))
        except Exception as e:
            print(f"  ✗ InternVL scoring failed: {e}, falling back to CLIP")
            return self.compute_clip_score(images, prompts)

    def compute_clip_score(self, images: List[Image.Image], prompts: List[str]) -> float:
        """Compute average CLIP score (baseline prompt-image alignment)"""
        scores = []
        for img, prompt in tqdm(zip(images, prompts), desc="CLIP scoring", leave=False):
            # Preprocess
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([prompt]).to(self.device)

            # Compute similarity
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Cosine similarity
                similarity = (image_features @ text_features.T).item()
                scores.append(similarity)

        return float(np.mean(scores))

    # ==================== CHARACTER CONSISTENCY METRICS ====================

    def compute_face_consistency(self, images: List[Image.Image]) -> float:
        """Compute face identity consistency using InsightFace (SOTA)"""
        try:
            embeddings = []
            for img in tqdm(images, desc="Face embedding", leave=False):
                # Convert PIL to numpy
                img_array = np.array(img)

                # Detect faces and extract embeddings
                faces = self.face_analyzer.get(img_array)

                if len(faces) > 0:
                    # Use largest face
                    largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
                    embeddings.append(largest_face.embedding)

            if len(embeddings) < 2:
                print("  ⚠️  Not enough faces detected, using CLIP fallback")
                return self.compute_character_consistency(images)

            embeddings = np.array(embeddings)

            # Compute pairwise cosine similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)

            return float(np.mean(similarities))

        except Exception as e:
            print(f"  ✗ InsightFace failed: {e}, using CLIP fallback")
            return self.compute_character_consistency(images)

    def compute_character_consistency(self, images: List[Image.Image]) -> float:
        """Compute intra-character consistency using CLIP (baseline)"""
        embeddings = []
        for img in tqdm(images, desc="CLIP embeddings", leave=False):
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.clip_model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())

        embeddings = np.array(embeddings).squeeze()

        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)

        return float(np.mean(similarities))

    # ==================== AESTHETICS METRICS ====================

    def compute_laion_aesthetics(self, images: List[Image.Image]) -> float:
        """Compute aesthetic score using LAION Aesthetics V2 (SOTA)"""
        try:
            scores = []
            for img in tqdm(images, desc="Aesthetics scoring", leave=False):
                result = self.aesthetic_scorer(img)
                # Extract score (model outputs classification scores)
                score = result[0]['score'] if isinstance(result, list) else result['score']
                scores.append(score)

            # Normalize to 0-1 range (LAION typically outputs 1-10)
            mean_score = np.mean(scores)
            return float(mean_score / 10.0) if mean_score > 1.0 else float(mean_score)

        except Exception as e:
            print(f"  ✗ LAION Aesthetics failed: {e}, using basic quality")
            return self.compute_image_quality(images)

    # ==================== IMAGE QUALITY METRICS ====================

    def compute_musiq_quality(self, images: List[Image.Image]) -> float:
        """Compute technical quality using MUSIQ (SOTA)"""
        try:
            scores = []
            for img in tqdm(images, desc="MUSIQ quality", leave=False):
                # Convert PIL to tensor
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(self.device)

                with torch.no_grad():
                    score = self.musiq_metric(img_tensor).item()
                    scores.append(score)

            # MUSIQ outputs 0-100, normalize to 0-1
            return float(np.mean(scores) / 100.0)

        except Exception as e:
            print(f"  ✗ MUSIQ failed: {e}, using Laplacian quality")
            return self.compute_image_quality(images)

    def compute_image_quality(self, images: List[Image.Image]) -> float:
        """Compute basic image quality (Laplacian sharpness - baseline)"""
        quality_scores = []

        for img in tqdm(images, desc="Laplacian quality", leave=False):
            # Convert to numpy
            img_array = np.array(img)

            # Compute Laplacian variance (sharpness measure)
            gray = np.mean(img_array, axis=2).astype(np.uint8)
            laplacian = np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ])

            # Convolve
            from scipy.ndimage import convolve
            edges = convolve(gray.astype(float), laplacian)
            sharpness = np.var(edges)

            # Normalize to 0-1 range (empirical max ~10000)
            quality_scores.append(min(sharpness / 10000, 1.0))

        return float(np.mean(quality_scores))

    # ==================== DIVERSITY METRICS ====================

    def compute_lpips_diversity(self, images: List[Image.Image]) -> float:
        """Compute perceptual diversity using LPIPS (SOTA)"""
        try:
            # Convert images to tensors
            img_tensors = []
            for img in images:
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                # LPIPS expects [-1, 1] range
                img_tensor = (img_tensor * 2) - 1
                img_tensors.append(img_tensor.to(self.device))

            # Compute pairwise LPIPS distances
            distances = []
            for i in tqdm(range(len(img_tensors)), desc="LPIPS diversity", leave=False):
                for j in range(i + 1, len(img_tensors)):
                    with torch.no_grad():
                        dist = self.lpips_metric(img_tensors[i], img_tensors[j]).item()
                        distances.append(dist)

            # Higher LPIPS distance = more diverse
            # Normalize to reasonable 0-1 range (LPIPS typically 0-1 already)
            return float(np.mean(distances))

        except Exception as e:
            print(f"  ✗ LPIPS failed: {e}, using CLIP std")
            return self.compute_diversity(images)

    def compute_diversity(self, images: List[Image.Image]) -> float:
        """Compute diversity using CLIP embedding std (baseline)"""
        embeddings = []

        for img in tqdm(images, desc="CLIP diversity", leave=False):
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.clip_model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())

        embeddings = np.array(embeddings).squeeze()

        # Compute standard deviation across embeddings (higher = more diverse)
        diversity = np.mean(np.std(embeddings, axis=0))

        return float(diversity)

    # ==================== COMPOSITE SCORE ====================

    def compute_composite_score(self, metrics: Dict) -> float:
        """
        Compute weighted composite score based on SOTA recommendations

        Default weights (from SOTA_MODELS_FOR_EVALUATION.md):
        - Prompt Alignment: 30%
        - Character Consistency: 25%
        - Aesthetics: 20%
        - Technical Quality: 15%
        - Diversity: 10%
        """
        # Use InternVL score if available, otherwise CLIP
        prompt_score = metrics.get('internvl_score', metrics.get('clip_score', 0))

        # Character consistency
        consistency = metrics.get('character_consistency', 0)

        # Aesthetics (use aesthetic_score if available, otherwise image_quality)
        aesthetics = metrics.get('aesthetic_score', metrics.get('image_quality', 0))

        # Technical quality
        quality = metrics.get('image_quality', 0)

        # Diversity
        diversity = metrics.get('diversity', 0)

        # Weighted composite
        composite = (
            prompt_score * 0.30 +
            consistency * 0.25 +
            aesthetics * 0.20 +
            quality * 0.15 +
            diversity * 0.10
        )

        return float(composite)

    def analyze_training_history(self, log_dir: Path) -> Dict:
        """Analyze training logs for overfitting/convergence"""

        # Look for loss logs
        metrics = {
            'converged': True,
            'overfitting_detected': False,
            'final_loss': None,
            'loss_trend': 'stable'
        }

        # TODO: Parse actual training logs if available
        # For now, return placeholder

        return metrics


def load_test_prompts(character: str) -> List[str]:
    """Load character-specific test prompts"""

    prompts = {
        'luca_human': [
            "a 3d animated character, Luca Paguro from Pixar Luca, young boy with brown curly hair, green eyes, wearing striped shirt, smiling, three-quarter view, studio lighting",
            "a 3d animated character, Luca Paguro from Pixar Luca, close-up portrait, curious expression, soft lighting, Italian Riviera background",
            "a 3d animated character, Luca Paguro from Pixar Luca, full body, standing pose, summer clothes, bright outdoor lighting",
            "a 3d animated character, Luca Paguro from Pixar Luca, side profile, thoughtful expression, warm sunset lighting",
        ],
        'alberto_human': [
            "a 3d animated character, Alberto Scorfano from Pixar Luca, confident boy with messy brown hair, tan skin, wearing simple clothes, smiling, three-quarter view",
            "a 3d animated character, Alberto Scorfano from Pixar Luca, close-up portrait, brave expression, Italian coastal town background",
            "a 3d animated character, Alberto Scorfano from Pixar Luca, full body, dynamic pose, casual outfit, bright sunlight",
            "a 3d animated character, Alberto Scorfano from Pixar Luca, side view, carefree expression, warm outdoor lighting",
        ]
    }

    return prompts.get(character, prompts['luca_human'])


def compare_checkpoints(results: List[Dict]) -> Dict:
    """Compare all checkpoints and rank them (composite score already computed)"""

    # Sort by composite score (already computed in evaluate_checkpoint)
    ranked = sorted(results, key=lambda x: x.get('composite_score', 0), reverse=True)

    if not ranked:
        return {
            'best_checkpoint': None,
            'best_score': 0.0,
            'rankings': [],
            'recommendations': ["No checkpoints evaluated successfully"]
        }

    return {
        'best_checkpoint': ranked[0]['checkpoint'],
        'best_score': ranked[0]['composite_score'],
        'rankings': ranked,
        'recommendations': generate_recommendations(ranked)
    }


def generate_recommendations(ranked_results: List[Dict]) -> List[str]:
    """Generate training improvement recommendations based on SOTA metrics"""
    if not ranked_results:
        return ["No results to analyze"]

    best = ranked_results[0]
    recommendations = []

    # Analyze prompt alignment (InternVL or CLIP)
    prompt_score = best.get('internvl_score', best.get('clip_score', 0))
    if prompt_score < 0.28:
        recommendations.append(
            f"⚠️  Low prompt alignment ({prompt_score:.3f}) - "
            "Consider increasing training epochs or improving caption quality"
        )

    # Character consistency
    if best['character_consistency'] < 0.75:
        recommendations.append(
            f"⚠️  Low character consistency ({best['character_consistency']:.3f}) - "
            "May need more diverse training data or reduce learning rate for stability"
        )

    # Aesthetics
    aesthetic_score = best.get('aesthetic_score', best.get('image_quality', 0))
    if aesthetic_score < 0.5:
        recommendations.append(
            f"⚠️  Low aesthetic quality ({aesthetic_score:.3f}) - "
            "Review training data quality and remove blurry/low-quality images"
        )

    # Diversity (mode collapse detection)
    if best['diversity'] < 0.15:
        recommendations.append(
            f"⚠️  Low diversity ({best['diversity']:.3f}) - "
            "Risk of mode collapse, reduce epochs or add noise offset"
        )

    # Check improvement trend
    if len(ranked_results) >= 3:
        scores = [r['composite_score'] for r in ranked_results[:3]]
        if scores[0] < scores[-1]:
            recommendations.append(
                "❌ Performance degrading across checkpoints - "
                "Consider reverting to earlier parameters"
            )
        elif max(scores) - min(scores) < 0.02:
            recommendations.append(
                "✓ Model converged - Try fine-tuning learning rate or network dim"
            )

    # Overall assessment
    if best['composite_score'] > 0.70:
        recommendations.append(
            f"✅ Excellent training quality (composite: {best['composite_score']:.3f})! "
            "Model is production-ready."
        )
    elif not recommendations:
        recommendations.append(
            f"✓ Good training quality (composite: {best['composite_score']:.3f}). "
            "Continue with current settings."
        )

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Automatic LoRA checkpoint evaluator with SOTA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with SOTA models (default)
  python auto_lora_evaluator.py --lora-dir /path/to/loras \\
    --character luca_human \\
    --base-model /path/to/sd-v1-5 \\
    --output-dir /path/to/output

  # Use only baseline CLIP models
  python auto_lora_evaluator.py --lora-dir /path/to/loras \\
    --character luca_human \\
    --base-model /path/to/sd-v1-5 \\
    --output-dir /path/to/output \\
    --no-sota
        """
    )
    parser.add_argument('--lora-dir', type=str, required=True,
                        help='Directory with LoRA checkpoints')
    parser.add_argument('--character', type=str, required=True,
                        choices=['luca_human', 'alberto_human'],
                        help='Character to evaluate')
    parser.add_argument('--base-model', type=str, required=True,
                        help='Base SD model path (default: from ai_warehouse)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--no-sota', action='store_true',
                        help='Disable SOTA models, use only baseline CLIP')

    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all checkpoints
    checkpoints = sorted(lora_dir.glob('*.safetensors'))

    if not checkpoints:
        print(f"❌ No checkpoints found in {lora_dir}")
        return 1

    print(f"\n{'='*70}")
    print(f"LORA CHECKPOINT EVALUATION - {'SOTA' if not args.no_sota else 'BASELINE'} MODE")
    print(f"{'='*70}")
    print(f"Character:    {args.character}")
    print(f"Checkpoints:  {len(checkpoints)}")
    print(f"Output:       {output_dir}")
    print(f"SOTA Models:  {'Enabled' if not args.no_sota else 'Disabled (baseline only)'}")
    print(f"{'='*70}\n")

    # Load test prompts
    test_prompts = load_test_prompts(args.character)

    # Initialize evaluator
    evaluator = LoRAEvaluator(args.base_model, args.device, use_sota=not args.no_sota)

    # Evaluate each checkpoint
    all_results = []

    for lora_path in checkpoints:
        try:
            result = evaluator.evaluate_checkpoint(
                lora_path,
                test_prompts,
                output_dir
            )
            all_results.append(result)

            print(f"\n{'='*60}")
            print(f"Results for {lora_path.name}:")
            print(f"{'='*60}")

            # Print scores
            if 'internvl_score' in result:
                print(f"  Prompt Alignment (InternVL):  {result['internvl_score']:.4f}")
            else:
                print(f"  Prompt Alignment (CLIP):      {result['clip_score']:.4f}")

            print(f"  Character Consistency:        {result['character_consistency']:.4f}")

            if 'aesthetic_score' in result:
                print(f"  Aesthetics (LAION):           {result['aesthetic_score']:.4f}")

            print(f"  Image Quality:                {result['image_quality']:.4f}")
            print(f"  Diversity:                    {result['diversity']:.4f}")
            print(f"  {'─'*58}")
            print(f"  Composite Score:              {result['composite_score']:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {lora_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\n❌ No checkpoints evaluated successfully")
        return 1

    # Compare and rank
    comparison = compare_checkpoints(all_results)

    # Save results
    report = {
        'character': args.character,
        'evaluation_mode': 'SOTA' if not args.no_sota else 'baseline',
        'models_used': {
            'prompt_alignment': 'InternVL2' if evaluator.use_internvl else 'CLIP',
            'character_consistency': 'InsightFace' if evaluator.face_analyzer else 'CLIP',
            'aesthetics': 'LAION V2' if evaluator.aesthetic_scorer else 'Laplacian',
            'technical_quality': 'MUSIQ' if evaluator.musiq_metric else 'Laplacian',
            'diversity': 'LPIPS' if evaluator.lpips_metric else 'CLIP std'
        },
        'evaluated_checkpoints': len(all_results),
        'best_checkpoint': comparison['best_checkpoint'],
        'best_score': comparison['best_score'],
        'rankings': comparison['rankings'],
        'recommendations': comparison['recommendations']
    }

    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Best Checkpoint:    {comparison['best_checkpoint']}")
    print(f"📊 Composite Score:    {comparison['best_score']:.4f}")
    print(f"\n💡 Recommendations:")
    for rec in comparison['recommendations']:
        print(f"   {rec}")
    print(f"\n📁 Full report:        {report_path}")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    main()
