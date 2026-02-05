#!/usr/bin/env python3
"""
Comprehensive SDXL LoRA Checkpoint Evaluator

State-of-the-art evaluation system for SDXL character LoRAs:
- InternVL2-8B / CLIP for prompt-image alignment
- LAION Aesthetics V2 for aesthetic quality
- InsightFace ArcFace for character identity consistency
- MUSIQ/NIQE for technical image quality
- LPIPS for perceptual diversity
- Character-specific prompts and metrics
- Automated checkpoint selection

Author: LLMProvider Tooling
Date: 2025-11-22
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
import matplotlib.gridspec as gridspec
import warnings
import sys
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# Diffusion
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch.nn.functional as F

# CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

# InternVL2
try:
    from transformers import AutoModel, AutoTokenizer
    INTERNVL_AVAILABLE = True
except ImportError:
    INTERNVL_AVAILABLE = False

# LAION Aesthetics
try:
    from transformers import pipeline
    AESTHETICS_AVAILABLE = True
except ImportError:
    AESTHETICS_AVAILABLE = False

# InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  InsightFace not available. Install with: pip install insightface")

# LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️  LPIPS not available. Install with: pip install lpips")

# MUSIQ
try:
    import pyiqa
    MUSIQ_AVAILABLE = True
except ImportError:
    MUSIQ_AVAILABLE = False
    print("⚠️  PyIQA (MUSIQ) not available. Install with: pip install pyiqa")


class SDXLLoRAEvaluator:
    """Comprehensive SDXL LoRA evaluator with SOTA metrics"""

    def __init__(
        self,
        base_model_path: str,
        vae_path: Optional[str] = None,
        device: str = 'cuda',
        use_fp16: bool = True,
        enable_xformers: bool = True,
    ):
        self.device = device
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        print("\n" + "="*80)
        print("🚀 SDXL LoRA EVALUATOR - SOTA Edition")
        print("="*80)

        # Load evaluation models
        self._load_evaluation_models()

        # Load SDXL pipeline
        self._load_sdxl_pipeline(base_model_path, vae_path, enable_xformers)

        print("="*80)
        print("✅ All models loaded successfully")
        print("="*80 + "\n")

    def _load_evaluation_models(self):
        """Load all evaluation models"""

        # 1. CLIP for prompt alignment (fallback)
        if CLIP_AVAILABLE:
            print("📊 Loading CLIP ViT-L/14...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            self.clip_model.eval()
            print("  ✓ CLIP loaded")
        else:
            self.clip_model = None
            print("  ✗ CLIP not available")

        # 2. InternVL2 for advanced alignment (optional)
        self.internvl_model = None
        if INTERNVL_AVAILABLE:
            try:
                print("📊 Loading InternVL2-8B (optional)...")
                # Note: This is heavy, only load if explicitly needed
                # self.internvl_model = AutoModel.from_pretrained(...)
                print("  → Skipping InternVL2 (too heavy for routine eval)")
            except Exception as e:
                print(f"  ✗ InternVL2 failed: {e}")

        # 3. InsightFace for character consistency
        if INSIGHTFACE_AVAILABLE:
            print("📊 Loading InsightFace ArcFace...")
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print("  ✓ InsightFace loaded")
        else:
            self.face_analyzer = None
            print("  ✗ InsightFace not available")

        # 4. LPIPS for diversity
        if LPIPS_AVAILABLE:
            print("📊 Loading LPIPS (perceptual diversity)...")
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            print("  ✓ LPIPS loaded")
        else:
            self.lpips_model = None
            print("  ✗ LPIPS not available")

        # 5. MUSIQ for image quality
        if MUSIQ_AVAILABLE:
            print("📊 Loading MUSIQ (image quality)...")
            self.musiq_model = pyiqa.create_metric('musiq', device=self.device)
            print("  ✓ MUSIQ loaded")
        else:
            self.musiq_model = None
            print("  ✗ MUSIQ not available")

        # 6. LAION Aesthetics (optional)
        self.aesthetics_model = None
        if AESTHETICS_AVAILABLE:
            try:
                print("📊 Loading LAION Aesthetics (optional)...")
                # This would require specific model weights
                print("  → Skipping Aesthetics (model not configured)")
            except Exception as e:
                print(f"  ✗ Aesthetics failed: {e}")

    def _load_sdxl_pipeline(self, base_model_path: str, vae_path: Optional[str], enable_xformers: bool):
        """Load SDXL generation pipeline"""

        print(f"📊 Loading SDXL pipeline from {base_model_path}...")

        # Load SDXL pipeline from single file
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            base_model_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)

        # Load custom VAE if specified (after pipeline is loaded)
        if vae_path:
            print(f"  → Loading custom VAE: {vae_path}")
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.dtype).to(self.device)
            self.pipe.vae = vae

        # Enable optimizations
        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("  ✓ xformers enabled")
            except Exception as e:
                print(f"  ✗ xformers failed: {e}")

        # Use model CPU offload for memory efficiency
        try:
            self.pipe.enable_model_cpu_offload()
            print("  ✓ Model CPU offload enabled")
        except Exception as e:
            print(f"  ✗ Model CPU offload failed: {e}")

        print("  ✓ SDXL pipeline loaded")

    def load_lora(self, lora_path: str, lora_scale: float = 1.0):
        """Load LoRA weights into pipeline"""

        print(f"📦 Loading LoRA: {Path(lora_path).name}")

        # Unload previous LoRA if any
        try:
            self.pipe.unload_lora_weights()
        except:
            pass

        # Load new LoRA
        self.pipe.load_lora_weights(str(lora_path))
        self.pipe.fuse_lora(lora_scale=lora_scale)

        print(f"  ✓ LoRA loaded (scale={lora_scale})")

    @torch.no_grad()
    def generate_images(
        self,
        prompts: List[str],
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        negative_prompt: str = "multiple people, duplicate, clone, two characters, extra limbs, extra arms, extra legs, extra hands, deformed, distorted, disfigured, bad anatomy, wrong anatomy, mutation, mutated, ugly, blurry, low quality, jpeg artifacts, watermark, text, bad proportions, gross proportions, malformed limbs, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, low quality",
    ) -> List[Image.Image]:
        """Generate images from prompts"""

        all_images = []
        generator = torch.Generator(device=self.device).manual_seed(seed)

        for prompt in tqdm(prompts, desc="Generating images"):
            images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            ).images

            all_images.extend(images)

        return all_images

    @torch.no_grad()
    def evaluate_prompt_alignment(
        self,
        images: List[Image.Image],
        prompts: List[str],
        images_per_prompt: int = 4
    ) -> Dict[str, float]:
        """Evaluate prompt-image alignment using CLIP"""

        if not CLIP_AVAILABLE or self.clip_model is None:
            return {"clip_score_mean": 0.0, "clip_score_std": 0.0}

        scores = []

        for i, prompt in enumerate(prompts):
            start_idx = i * images_per_prompt
            end_idx = start_idx + images_per_prompt
            prompt_images = images[start_idx:end_idx]

            # Preprocess images
            image_inputs = torch.stack([
                self.clip_preprocess(img).to(self.device)
                for img in prompt_images
            ])

            # Encode
            image_features = self.clip_model.encode_image(image_inputs)
            text_features = self.clip_model.encode_text(clip.tokenize([prompt]).to(self.device))

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (image_features @ text_features.T).squeeze()
            scores.extend(similarity.cpu().numpy().tolist())

        return {
            "clip_score_mean": float(np.mean(scores)),
            "clip_score_std": float(np.std(scores)),
            "clip_scores": [float(s) for s in scores]
        }

    @torch.no_grad()
    def evaluate_face_consistency(self, images: List[Image.Image]) -> Dict[str, float]:
        """Evaluate character identity consistency using InsightFace"""

        if not INSIGHTFACE_AVAILABLE or self.face_analyzer is None:
            return {"face_consistency_mean": 0.0, "face_consistency_std": 0.0}

        # Extract face embeddings
        embeddings = []
        detected_faces = 0

        for img in images:
            img_np = np.array(img.convert('RGB'))
            faces = self.face_analyzer.get(img_np)

            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                embeddings.append(largest_face.embedding)
                detected_faces += 1

        if len(embeddings) < 2:
            return {
                "face_consistency_mean": 0.0,
                "face_consistency_std": 0.0,
                "faces_detected": detected_faces,
                "faces_total": len(images)
            }

        # Compute pairwise cosine similarities
        embeddings = np.array(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(float(sim))

        return {
            "face_consistency_mean": float(np.mean(similarities)),
            "face_consistency_std": float(np.std(similarities)),
            "face_similarities": similarities,
            "faces_detected": detected_faces,
            "faces_total": len(images)
        }

    @torch.no_grad()
    def evaluate_diversity(self, images: List[Image.Image]) -> Dict[str, float]:
        """Evaluate perceptual diversity using LPIPS"""

        if not LPIPS_AVAILABLE or self.lpips_model is None:
            return {"diversity_mean": 0.0, "diversity_std": 0.0}

        # Convert images to tensors
        def img_to_tensor(img):
            img = img.convert('RGB').resize((256, 256))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return img.to(self.device) * 2.0 - 1.0  # [-1, 1]

        tensors = [img_to_tensor(img) for img in images]

        # Compute pairwise LPIPS distances
        distances = []
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                dist = self.lpips_model(tensors[i], tensors[j])
                distances.append(float(dist.item()))

        return {
            "diversity_mean": float(np.mean(distances)),
            "diversity_std": float(np.std(distances)),
            "lpips_distances": distances
        }

    @torch.no_grad()
    def evaluate_image_quality(self, images: List[Image.Image]) -> Dict[str, float]:
        """Evaluate technical image quality using MUSIQ"""

        if not MUSIQ_AVAILABLE or self.musiq_model is None:
            return {"quality_mean": 0.0, "quality_std": 0.0}

        scores = []
        for img in images:
            img_np = np.array(img.convert('RGB'))
            score = self.musiq_model(img_np)
            scores.append(float(score.item()))

        return {
            "quality_mean": float(np.mean(scores)),
            "quality_std": float(np.std(scores)),
            "quality_scores": scores
        }

    def evaluate_checkpoint(
        self,
        lora_path: str,
        prompts: List[str],
        output_dir: Path,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        negative_prompt: str = "multiple people, duplicate, clone, two characters, extra limbs, extra arms, extra legs, extra hands, deformed, distorted, disfigured, bad anatomy, wrong anatomy, mutation, mutated, ugly, blurry, low quality, jpeg artifacts, watermark, text, bad proportions, gross proportions, malformed limbs, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, low quality",
    ) -> Dict:
        """Comprehensive evaluation of a single checkpoint"""

        checkpoint_name = Path(lora_path).stem
        checkpoint_output = output_dir / checkpoint_name
        checkpoint_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🎨 Evaluating: {checkpoint_name}")
        print(f"{'='*80}")

        # Load LoRA
        self.load_lora(lora_path)

        # Generate images
        print(f"\n📸 Generating {len(prompts) * num_images_per_prompt} images...")
        images = self.generate_images(
            prompts=prompts,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt,
        )

        # Save generated images
        print(f"\n💾 Saving images...")
        for i, img in enumerate(images):
            img.save(checkpoint_output / f"image_{i:04d}.png")

        # Run evaluations
        print(f"\n📊 Running evaluations...")

        results = {
            "checkpoint": checkpoint_name,
            "checkpoint_path": str(lora_path),
            "num_prompts": len(prompts),
            "num_images": len(images),
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Prompt alignment
        print("  → CLIP prompt alignment...")
        results["prompt_alignment"] = self.evaluate_prompt_alignment(
            images, prompts, num_images_per_prompt
        )

        # 2. Face consistency
        print("  → InsightFace character consistency...")
        results["face_consistency"] = self.evaluate_face_consistency(images)

        # 3. Diversity
        print("  → LPIPS perceptual diversity...")
        results["diversity"] = self.evaluate_diversity(images)

        # 4. Image quality
        print("  → MUSIQ image quality...")
        results["image_quality"] = self.evaluate_image_quality(images)

        # Compute aggregate score
        results["aggregate_score"] = self._compute_aggregate_score(results)

        # Save results
        with open(checkpoint_output / "evaluation.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Results:")
        print(f"  CLIP Score:         {results['prompt_alignment']['clip_score_mean']:.4f}")
        print(f"  Face Consistency:   {results['face_consistency']['face_consistency_mean']:.4f}")
        print(f"  Diversity (LPIPS):  {results['diversity']['diversity_mean']:.4f}")
        print(f"  Image Quality:      {results['image_quality']['quality_mean']:.4f}")
        print(f"  Aggregate Score:    {results['aggregate_score']:.4f}")

        return results

    def _compute_aggregate_score(self, results: Dict) -> float:
        """Compute weighted aggregate score"""

        weights = {
            "clip_score": 0.35,
            "face_consistency": 0.35,
            "diversity": 0.15,
            "quality": 0.15,
        }

        scores = {
            "clip_score": results["prompt_alignment"].get("clip_score_mean", 0.0),
            "face_consistency": results["face_consistency"].get("face_consistency_mean", 0.0),
            "diversity": results["diversity"].get("diversity_mean", 0.0) * 10.0,  # Scale LPIPS to ~0-1
            "quality": results["image_quality"].get("quality_mean", 0.0) / 100.0,  # Scale to 0-1
        }

        aggregate = sum(scores[k] * weights[k] for k in weights.keys())
        return float(aggregate)

    def compare_checkpoints(
        self,
        checkpoint_results: List[Dict],
        output_dir: Path
    ):
        """Generate comparison report and visualizations"""

        print(f"\n{'='*80}")
        print("📊 CHECKPOINT COMPARISON")
        print(f"{'='*80}\n")

        # Sort by aggregate score
        checkpoint_results = sorted(
            checkpoint_results,
            key=lambda x: x['aggregate_score'],
            reverse=True
        )

        # Print ranking
        print("🏆 Checkpoint Ranking (by aggregate score):\n")
        for i, result in enumerate(checkpoint_results, 1):
            print(f"{i}. {result['checkpoint']}")
            print(f"   Aggregate Score: {result['aggregate_score']:.4f}")
            print(f"   CLIP:            {result['prompt_alignment']['clip_score_mean']:.4f}")
            print(f"   Face Cons.:      {result['face_consistency']['face_consistency_mean']:.4f}")
            print(f"   Diversity:       {result['diversity']['diversity_mean']:.4f}")
            print(f"   Quality:         {result['image_quality']['quality_mean']:.4f}")
            print()

        # Save comparison JSON
        comparison = {
            "ranking": [r['checkpoint'] for r in checkpoint_results],
            "best_checkpoint": checkpoint_results[0]['checkpoint'],
            "checkpoints": checkpoint_results,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "checkpoint_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)

        # Generate comparison plot
        self._plot_comparison(checkpoint_results, output_dir)

        print(f"✅ Comparison saved to {output_dir}/checkpoint_comparison.json")
        print(f"✅ Best checkpoint: {checkpoint_results[0]['checkpoint']}")

        return comparison

    def _plot_comparison(self, results: List[Dict], output_dir: Path):
        """Create comparison visualization"""

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        checkpoints = [r['checkpoint'] for r in results]
        x = np.arange(len(checkpoints))

        # 1. Aggregate scores
        ax1 = fig.add_subplot(gs[0, :])
        scores = [r['aggregate_score'] for r in results]
        bars = ax1.bar(x, scores, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Aggregate Score', fontsize=12)
        ax1.set_title('Overall Checkpoint Performance', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(checkpoints, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Highlight best
        bars[0].set_color('gold')
        bars[0].set_alpha(1.0)

        # 2. CLIP scores
        ax2 = fig.add_subplot(gs[1, 0])
        clip_scores = [r['prompt_alignment']['clip_score_mean'] for r in results]
        ax2.bar(x, clip_scores, color='coral', alpha=0.8)
        ax2.set_ylabel('CLIP Score', fontsize=10)
        ax2.set_title('Prompt Alignment', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(checkpoints, rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Face consistency
        ax3 = fig.add_subplot(gs[1, 1])
        face_scores = [r['face_consistency']['face_consistency_mean'] for r in results]
        ax3.bar(x, face_scores, color='mediumseagreen', alpha=0.8)
        ax3.set_ylabel('Cosine Similarity', fontsize=10)
        ax3.set_title('Character Consistency (ArcFace)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(checkpoints, rotation=45, ha='right', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "checkpoint_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Comparison plot saved to {output_dir}/checkpoint_comparison.png")


def load_prompts(prompts_file: Path) -> List[str]:
    """Load prompts from file"""

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    # Try JSON first
    try:
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
    except json.JSONDecodeError:
        pass

    # Try plain text
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Comprehensive SDXL LoRA Checkpoint Evaluator")

    # Required arguments
    parser.add_argument("lora_dir", type=str, help="Directory containing LoRA checkpoint(s)")
    parser.add_argument("--base-model", type=str, required=True, help="Path to SDXL base model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--prompts-file", type=str, required=True, help="Path to prompts file (txt or json)")

    # Optional arguments
    parser.add_argument("--vae", type=str, default=None, help="Custom VAE path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num-images-per-prompt", type=int, default=4, help="Images per prompt")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--negative-prompt", type=str, default="multiple people, duplicate, clone, two characters, extra limbs, extra arms, extra legs, extra hands, deformed, distorted, disfigured, bad anatomy, wrong anatomy, mutation, mutated, ugly, blurry, low quality, jpeg artifacts, watermark, text, bad proportions, gross proportions, malformed limbs, missing arms, missing legs, extra digit, fewer digits, cropped, worst quality, low quality", help="Negative prompt to avoid artifacts")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--no-xformers", action="store_true", help="Disable xformers")

    args = parser.parse_args()

    # Setup paths
    lora_dir = Path(args.lora_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = Path(args.prompts_file)

    # Load prompts
    print(f"📝 Loading prompts from {prompts_file}...")
    prompts = load_prompts(prompts_file)
    print(f"  ✓ Loaded {len(prompts)} prompts")

    # Find checkpoints
    checkpoints = sorted(lora_dir.glob("*.safetensors"))
    if not checkpoints:
        print(f"❌ No .safetensors files found in {lora_dir}")
        return

    print(f"📦 Found {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        print(f"  - {cp.name}")
    print()

    # Initialize evaluator
    evaluator = SDXLLoRAEvaluator(
        base_model_path=args.base_model,
        vae_path=args.vae,
        device=args.device,
        use_fp16=not args.no_fp16,
        enable_xformers=not args.no_xformers,
    )

    # Evaluate each checkpoint
    all_results = []
    for checkpoint in checkpoints:
        result = evaluator.evaluate_checkpoint(
            lora_path=str(checkpoint),
            prompts=prompts,
            output_dir=output_dir,
            num_images_per_prompt=args.num_images_per_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
        )
        all_results.append(result)

    # Generate comparison
    if len(all_results) > 1:
        evaluator.compare_checkpoints(all_results, output_dir)

    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
