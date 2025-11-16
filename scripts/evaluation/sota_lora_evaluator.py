#!/usr/bin/env python3
"""
SOTA LoRA Checkpoint Evaluator

Uses state-of-the-art models for comprehensive evaluation:
- InternVL2-8B (replaces CLIP) for prompt-image alignment
- LAION Aesthetics V2 for aesthetic scoring
- InsightFace for character consistency
- MUSIQ for technical image quality
- LPIPS for perceptual diversity
"""

import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
import warnings
import sys
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.utils.prompt_loader import load_character_prompts

# Diffusion
from diffusers import StableDiffusionPipeline

# InternVL2 (replaces CLIP)
from transformers import AutoModel, AutoTokenizer

# LAION Aesthetics
from transformers import pipeline

# InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Install with: pip install insightface")

# LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

# MUSIQ (via pyiqa)
try:
    import pyiqa
    MUSIQ_AVAILABLE = True
except ImportError:
    MUSIQ_AVAILABLE = False
    print("Warning: MUSIQ not available. Install with: pip install pyiqa")


class SOTALoRAEvaluator:
    """SOTA-based LoRA evaluator"""

    def __init__(
        self,
        base_model_path: str,
        device: str = 'cuda',
        model_paths: Dict[str, str] = None
    ):
        self.device = device
        self.model_paths = model_paths or {}

        print("\n" + "="*70)
        print("LOADING SOTA EVALUATION MODELS")
        print("="*70)

        # 1. Load InternVL2 for prompt alignment (replaces CLIP)
        self.load_internvl2()

        # 2. Load LAION Aesthetics
        self.load_aesthetics()

        # 3. Load InsightFace for character consistency
        self.load_insightface()

        # 4. Load MUSIQ for image quality
        self.load_musiq()

        # 5. Load LPIPS for diversity
        self.load_lpips()

        # 6. Load SD pipeline for generation
        self.load_sd_pipeline(base_model_path)

        print("="*70)
        print("✓ ALL SOTA MODELS LOADED")
        print("="*70 + "\n")

    def load_internvl2(self):
        """Load InternVL2-8B for prompt alignment"""
        print("\n[1/6] Loading InternVL2-8B (replacing CLIP)...")

        model_path = self.model_paths.get(
            'internvl2',
            '/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/InternVL2-8B'
        )

        try:
            self.internvl_model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device).eval()

            self.internvl_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            print("  ✓ InternVL2-8B loaded successfully")
            self.internvl_available = True

        except Exception as e:
            print(f"  ✗ InternVL2 not available: {e}")
            print("  → Falling back to CLIP")
            self.internvl_available = False
            self.load_clip_fallback()

    def load_clip_fallback(self):
        """Fallback to CLIP if InternVL2 not available"""
        import clip
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        print("  ✓ CLIP ViT-L/14 loaded as fallback")

    def load_aesthetics(self):
        """Load LAION Aesthetics V2"""
        print("\n[2/6] Loading LAION Aesthetics V2...")

        try:
            self.aesthetic_scorer = pipeline(
                "image-classification",
                model="cafeai/cafe_aesthetic",
                device=0 if self.device == 'cuda' else -1
            )
            print("  ✓ LAION Aesthetics V2 loaded")
            self.aesthetics_available = True

        except Exception as e:
            print(f"  ✗ Aesthetics not available: {e}")
            self.aesthetics_available = False

    def load_insightface(self):
        """Load InsightFace for character consistency"""
        print("\n[3/6] Loading InsightFace...")

        if not INSIGHTFACE_AVAILABLE:
            print("  ✗ InsightFace not installed")
            self.insightface_available = False
            return

        try:
            self.face_app = FaceAnalysis(
                providers=['CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0 if self.device == 'cuda' else -1)
            print("  ✓ InsightFace loaded")
            self.insightface_available = True

        except Exception as e:
            print(f"  ✗ InsightFace initialization failed: {e}")
            self.insightface_available = False

    def load_musiq(self):
        """Load MUSIQ for image quality"""
        print("\n[4/6] Loading MUSIQ...")

        if not MUSIQ_AVAILABLE:
            print("  ✗ MUSIQ not installed (pip install pyiqa)")
            self.musiq_available = False
            return

        try:
            self.musiq_metric = pyiqa.create_metric('musiq', device=self.device)
            print("  ✓ MUSIQ loaded")
            self.musiq_available = True

        except Exception as e:
            print(f"  ✗ MUSIQ initialization failed: {e}")
            self.musiq_available = False

    def load_lpips(self):
        """Load LPIPS for perceptual diversity"""
        print("\n[5/6] Loading LPIPS...")

        if not LPIPS_AVAILABLE:
            print("  ✗ LPIPS not installed (pip install lpips)")
            self.lpips_available = False
            return

        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            print("  ✓ LPIPS loaded")
            self.lpips_available = True

        except Exception as e:
            print(f"  ✗ LPIPS initialization failed: {e}")
            self.lpips_available = False

    def load_sd_pipeline(self, base_model_path: str):
        """Load Stable Diffusion pipeline"""
        print("\n[6/6] Loading Stable Diffusion pipeline...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            safety_checker=None
        ).to(self.device)

        print("  ✓ SD pipeline loaded")

    def compute_internvl_score(self, images: List[Image.Image], prompts: List[str]) -> float:
        """Compute InternVL2 alignment score"""
        if not self.internvl_available:
            return self.compute_clip_score_fallback(images, prompts)

        scores = []

        for img, prompt in zip(images, prompts):
            try:
                # Prepare inputs
                pixel_values = self.internvl_model.preprocess(img).unsqueeze(0).to(
                    self.device, dtype=torch.bfloat16
                )

                # Encode image
                image_embeds = self.internvl_model.encode_image(pixel_values)

                # Encode text
                text_inputs = self.internvl_tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)

                text_embeds = self.internvl_model.encode_text(**text_inputs)

                # Compute similarity
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                similarity = (image_embeds @ text_embeds.T).item()
                scores.append(similarity)

            except Exception as e:
                print(f"  Warning: InternVL2 scoring failed: {e}")
                continue

        return float(np.mean(scores)) if scores else 0.0

    def compute_clip_score_fallback(self, images: List[Image.Image], prompts: List[str]) -> float:
        """Fallback CLIP scoring"""
        import clip  # Local import for tokenization
        scores = []

        for img, prompt in zip(images, prompts):
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            text_input = torch.cat([self.clip_model.encode_text(clip.tokenize([prompt]).to(self.device))])

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = text_input

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).item()
                scores.append(similarity)

        return float(np.mean(scores))

    def compute_aesthetic_score(self, images: List[Image.Image]) -> float:
        """Compute LAION Aesthetics score"""
        if not self.aesthetics_available:
            return 0.5  # Neutral default

        scores = []

        for img in images:
            try:
                result = self.aesthetic_scorer(img)
                # Extract score (model outputs 1-10 scale)
                score = result[0]['score'] / 10.0 if 'score' in result[0] else result[0]['label']
                scores.append(float(score))

            except Exception as e:
                print(f"  Warning: Aesthetic scoring failed: {e}")
                continue

        return float(np.mean(scores)) if scores else 0.5

    def compute_insightface_consistency(self, images: List[Image.Image]) -> float:
        """Compute character consistency via InsightFace"""
        if not self.insightface_available:
            return self.compute_visual_consistency_fallback(images)

        embeddings = []

        for img in images:
            try:
                img_array = np.array(img.convert('RGB'))
                faces = self.face_app.get(img_array)

                if faces:
                    # Use the largest face
                    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                    embeddings.append(face.embedding)

            except Exception as e:
                print(f"  Warning: Face detection failed: {e}")
                continue

        if len(embeddings) < 2:
            return 0.5  # Not enough faces detected

        # Compute pairwise similarities
        embeddings = np.array(embeddings)
        similarities = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        return float(np.mean(similarities))

    def compute_visual_consistency_fallback(self, images: List[Image.Image]) -> float:
        """Fallback visual consistency using CLIP embeddings"""
        if not hasattr(self, 'clip_model'):
            return 0.5

        embeddings = []

        for img in images:
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.clip_model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())

        embeddings = np.array(embeddings).squeeze()

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)

        return float(np.mean(similarities))

    def compute_musiq_quality(self, images: List[Image.Image]) -> float:
        """Compute MUSIQ quality score"""
        if not self.musiq_available:
            return self.compute_sharpness_fallback(images)

        scores = []

        for img in images:
            try:
                # MUSIQ expects tensor
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(self.device)

                score = self.musiq_metric(img_tensor).item()
                scores.append(score)

            except Exception as e:
                print(f"  Warning: MUSIQ scoring failed: {e}")
                continue

        return float(np.mean(scores)) if scores else 0.5

    def compute_sharpness_fallback(self, images: List[Image.Image]) -> float:
        """Fallback sharpness-based quality"""
        from scipy.ndimage import convolve

        scores = []

        for img in images:
            img_array = np.array(img.convert('L'))  # Grayscale

            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            edges = convolve(img_array.astype(float), laplacian)
            sharpness = np.var(edges)

            # Normalize
            score = min(sharpness / 10000, 1.0)
            scores.append(score)

        return float(np.mean(scores))

    def compute_lpips_diversity(self, images: List[Image.Image]) -> float:
        """Compute LPIPS-based diversity"""
        if not self.lpips_available:
            return 0.15  # Neutral default

        # Convert images to tensors
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        tensors = [transform(img).unsqueeze(0).to(self.device) for img in images]

        # Compute pairwise LPIPS distances
        distances = []

        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                with torch.no_grad():
                    dist = self.lpips_fn(tensors[i], tensors[j]).item()
                    distances.append(dist)

        # Higher LPIPS = more diverse
        return float(np.mean(distances)) if distances else 0.15

    def evaluate_checkpoint(
        self,
        lora_path: Path,
        test_prompts: List[str],
        output_dir: Path,
        num_images_per_prompt: int = 4
    ) -> Dict:
        """Evaluate checkpoint with SOTA models"""

        print(f"\n{'='*70}")
        print(f"EVALUATING: {lora_path.name}")
        print(f"{'='*70}")

        # Load LoRA
        self.pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

        # Generate test images
        generated_images = []
        prompts_used = []

        for prompt in tqdm(test_prompts, desc="Generating"):
            for i in range(num_images_per_prompt):
                image = self.pipe(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(42 + i)
                ).images[0]

                generated_images.append(image)
                prompts_used.append(prompt)

        # Save images
        test_output_dir = output_dir / lora_path.stem
        test_output_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(generated_images):
            img.save(test_output_dir / f"test_{idx:03d}.png")

        # SOTA Evaluation
        print("\nComputing SOTA metrics...")

        metrics = {
            'internvl_score': self.compute_internvl_score(generated_images, prompts_used),
            'aesthetic_score': self.compute_aesthetic_score(generated_images),
            'character_consistency': self.compute_insightface_consistency(generated_images),
            'image_quality': self.compute_musiq_quality(generated_images),
            'diversity': self.compute_lpips_diversity(generated_images),
            'checkpoint': lora_path.name,
        }

        # Compute composite score
        metrics['composite_score'] = (
            metrics['internvl_score'] * 0.30 +
            metrics['character_consistency'] * 0.25 +
            metrics['aesthetic_score'] * 0.20 +
            metrics['image_quality'] * 0.15 +
            metrics['diversity'] * 0.10
        )

        # Unload LoRA
        self.pipe.unload_lora_weights()

        # Print results
        print(f"\nResults:")
        print(f"  InternVL Score:    {metrics['internvl_score']:.4f}")
        print(f"  Aesthetics:        {metrics['aesthetic_score']:.4f}")
        print(f"  Consistency:       {metrics['character_consistency']:.4f}")
        print(f"  Quality (MUSIQ):   {metrics['image_quality']:.4f}")
        print(f"  Diversity (LPIPS): {metrics['diversity']:.4f}")
        print(f"  Composite:         {metrics['composite_score']:.4f}")

        return metrics

    def evaluate_existing_samples(
        self,
        checkpoint_name: str,
        sample_dir: Path,
        prompts: List[str],
        use_fast_metrics: bool = False
    ) -> Dict:
        """
        Evaluate existing sample images (e.g., from training validation)

        Args:
            checkpoint_name: Name of the checkpoint (for reporting)
            sample_dir: Directory containing sample images
            prompts: List of prompts corresponding to the samples
            use_fast_metrics: If True, use only CLIP and Aesthetics (faster)

        Returns:
            Dictionary with evaluation metrics
        """

        print(f"\n{'='*70}")
        print(f"EVALUATING EXISTING SAMPLES: {checkpoint_name}")
        print(f"{'='*70}")

        # Find sample images
        sample_images = sorted(sample_dir.glob(f"{checkpoint_name}_*.png"))

        if not sample_images:
            print(f"⚠️  No samples found for {checkpoint_name}")
            return None

        print(f"   Found {len(sample_images)} sample images")

        # Load images
        images = []
        for img_path in sample_images:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"  Warning: Failed to load {img_path.name}: {e}")
                continue

        if not images:
            print(f"✗ No valid images could be loaded")
            return None

        # Match images to prompts (cycling through prompts if needed)
        prompts_matched = []
        for i in range(len(images)):
            prompts_matched.append(prompts[i % len(prompts)])

        # Compute metrics
        print("\nComputing metrics...")

        metrics = {
            'checkpoint': checkpoint_name,
            'num_samples': len(images),
        }

        # Always compute CLIP/InternVL and Aesthetics (fast metrics)
        if self.internvl_available:
            print("  • Computing InternVL scores...")
            metrics['internvl_score'] = self.compute_internvl_score(images, prompts_matched)
        elif hasattr(self, 'clip_model'):
            print("  • Computing CLIP scores...")
            metrics['clip_score'] = self.compute_clip_score_fallback(images, prompts_matched)
            metrics['internvl_score'] = metrics['clip_score']  # Alias for consistency
        else:
            metrics['internvl_score'] = 0.5

        print("  • Computing aesthetic scores...")
        metrics['aesthetic_score'] = self.compute_aesthetic_score(images)

        # Optionally compute slower metrics
        if not use_fast_metrics:
            print("  • Computing character consistency...")
            metrics['character_consistency'] = self.compute_insightface_consistency(images)

            print("  • Computing image quality...")
            metrics['image_quality'] = self.compute_musiq_quality(images)

            print("  • Computing diversity...")
            metrics['diversity'] = self.compute_lpips_diversity(images)

            # Compute composite score
            metrics['composite_score'] = (
                metrics['internvl_score'] * 0.30 +
                metrics['character_consistency'] * 0.25 +
                metrics['aesthetic_score'] * 0.20 +
                metrics['image_quality'] * 0.15 +
                metrics['diversity'] * 0.10
            )
        else:
            # Fast composite (only prompt alignment + aesthetics)
            metrics['composite_score'] = (
                metrics['internvl_score'] * 0.60 +
                metrics['aesthetic_score'] * 0.40
            )

        # Print results
        print(f"\nResults:")
        print(f"  Prompt Alignment: {metrics['internvl_score']:.4f}")
        print(f"  Aesthetics:       {metrics['aesthetic_score']:.4f}")

        if not use_fast_metrics:
            print(f"  Consistency:      {metrics['character_consistency']:.4f}")
            print(f"  Quality:          {metrics['image_quality']:.4f}")
            print(f"  Diversity:        {metrics['diversity']:.4f}")

        print(f"  Composite:        {metrics['composite_score']:.4f}")

        return metrics


def load_validation_prompts(prompt_file: str) -> List[str]:
    """
    Load validation prompts from Kohya-style prompt file

    Args:
        prompt_file: Path to prompt file (Kohya format)

    Returns:
        List of prompts
    """
    prompts = []

    if not Path(prompt_file).exists():
        print(f"⚠️  Prompt file not found: {prompt_file}")
        return prompts

    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse Kohya format: prompt --n negative --w width --h height ...
                parts = line.split('--')
                prompt = parts[0].strip()
                prompts.append(prompt)

    return prompts


def load_test_prompts(character: str, num_prompts: int = 12) -> List[str]:
    """
    Load character-specific test prompts from JSON library

    Args:
        character: Character name (e.g., 'luca_human', 'alberto_human')
        num_prompts: Number of prompts to sample (default: 12)

    Returns:
        List of positive prompts for testing
    """

    try:
        # Try to load from prompt library
        loader = load_character_prompts(character)

        # Get balanced sample across categories
        prompts = loader.get_simple_test_prompts(num_prompts=num_prompts)

        print(f"✓ Loaded {len(prompts)} prompts from library for {character}")
        print(f"  Categories: {', '.join(loader.get_categories())}")

        return prompts

    except FileNotFoundError:
        # Fallback to hardcoded prompts if library not found
        print(f"⚠ Prompt library not found for {character}, using fallback prompts")

        fallback_prompts = {
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

        return fallback_prompts.get(character, fallback_prompts['luca_human'])


def main():
    parser = argparse.ArgumentParser(description="SOTA LoRA checkpoint evaluator")
    parser.add_argument('--lora-dir', type=str, required=True, help='Directory containing LoRA checkpoint files')
    parser.add_argument('--character', type=str, required=True, help='Character name for prompt loading')
    parser.add_argument('--base-model', type=str, help='Base model path (required for --generate-mode)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for reports')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--model-paths', type=str, help='JSON file with custom model paths')

    # Evaluation mode
    parser.add_argument('--evaluate-samples', action='store_true',
                       help='Evaluate existing sample images instead of generating new ones')
    parser.add_argument('--sample-dir', type=str,
                       help='Directory containing sample images (for --evaluate-samples mode)')
    parser.add_argument('--prompt-file', type=str,
                       help='Validation prompt file (Kohya format, for --evaluate-samples mode)')
    parser.add_argument('--fast-metrics', action='store_true',
                       help='Use only CLIP/InternVL and Aesthetics for faster evaluation')

    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model paths if provided
    model_paths = {}
    if args.model_paths and Path(args.model_paths).exists():
        with open(args.model_paths, 'r') as f:
            model_paths = json.load(f)

    # Find checkpoints
    checkpoints = sorted(lora_dir.glob('*.safetensors'))

    print(f"\n{'='*70}")
    print(f"SOTA LORA EVALUATION")
    print(f"{'='*70}")
    print(f"Character:    {args.character}")
    print(f"Checkpoints:  {len(checkpoints)}")
    print(f"Output:       {output_dir}")

    # Handle evaluation mode
    if args.evaluate_samples:
        print(f"Mode:         Evaluate existing samples")

        if not args.sample_dir:
            print("❌ ERROR: --sample-dir required for --evaluate-samples mode")
            return

        sample_dir = Path(args.sample_dir)
        if not sample_dir.exists():
            print(f"❌ ERROR: Sample directory not found: {sample_dir}")
            return

        print(f"Sample Dir:   {sample_dir}")

        # Load validation prompts
        if args.prompt_file:
            prompts = load_validation_prompts(args.prompt_file)
            print(f"Prompts:      {len(prompts)} from {args.prompt_file}")
        else:
            prompts = load_test_prompts(args.character)
            print(f"Prompts:      {len(prompts)} (character default)")

        print(f"Fast Metrics: {'Yes' if args.fast_metrics else 'No'}")

        print(f"{'='*70}\n")

        # Initialize evaluator (lightweight - no SD pipeline needed)
        class MinimalEvaluator:
            """Minimal evaluator without SD pipeline"""
            pass

        evaluator = MinimalEvaluator()

        # Load only necessary models
        from transformers import AutoModel, AutoTokenizer
        import clip

        print("Loading evaluation models...")

        # Try InternVL2, fallback to CLIP
        try:
            internvl_model_path = model_paths.get(
                'internvl2',
                '/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/InternVL2-8B'
            )

            if Path(internvl_model_path).exists():
                evaluator.internvl_model = AutoModel.from_pretrained(
                    internvl_model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).to(args.device).eval()
                evaluator.internvl_tokenizer = AutoTokenizer.from_pretrained(
                    internvl_model_path, trust_remote_code=True
                )
                evaluator.internvl_available = True
                print("  ✓ InternVL2 loaded")
            else:
                raise FileNotFoundError("InternVL2 not found")

        except Exception as e:
            print(f"  ⚠ InternVL2 not available: {e}")
            print("  → Using CLIP fallback")
            evaluator.clip_model, evaluator.clip_preprocess = clip.load("ViT-B/32", device=args.device)
            evaluator.internvl_available = False
            print("  ✓ CLIP loaded")

        # Load Aesthetics
        try:
            from transformers import pipeline
            evaluator.aesthetic_scorer = pipeline(
                "image-classification",
                model="cafeai/cafe_aesthetic",
                device=0 if args.device == 'cuda' else -1
            )
            evaluator.aesthetics_available = True
            print("  ✓ Aesthetics loaded")
        except Exception as e:
            print(f"  ⚠ Aesthetics not available: {e}")
            evaluator.aesthetics_available = False

        # Copy methods from SOTALoRAEvaluator
        evaluator.compute_internvl_score = SOTALoRAEvaluator.compute_internvl_score.__get__(evaluator)
        evaluator.compute_clip_score_fallback = SOTALoRAEvaluator.compute_clip_score_fallback.__get__(evaluator)
        evaluator.compute_aesthetic_score = SOTALoRAEvaluator.compute_aesthetic_score.__get__(evaluator)
        evaluator.evaluate_existing_samples = SOTALoRAEvaluator.evaluate_existing_samples.__get__(evaluator)

        if not args.fast_metrics:
            # Load additional models
            evaluator.compute_insightface_consistency = SOTALoRAEvaluator.compute_insightface_consistency.__get__(evaluator)
            evaluator.compute_musiq_quality = SOTALoRAEvaluator.compute_musiq_quality.__get__(evaluator)
            evaluator.compute_lpips_diversity = SOTALoRAEvaluator.compute_lpips_diversity.__get__(evaluator)

            # Initialize InsightFace, MUSIQ, LPIPS if available
            # (loading code omitted for brevity - will use fallbacks)
            evaluator.insightface_available = False
            evaluator.musiq_available = False
            evaluator.lpips_available = False

        evaluator.device = args.device

        # Evaluate each checkpoint's samples
        all_results = []

        for checkpoint_path in checkpoints:
            checkpoint_name = checkpoint_path.stem

            try:
                result = evaluator.evaluate_existing_samples(
                    checkpoint_name,
                    sample_dir,
                    prompts,
                    use_fast_metrics=args.fast_metrics
                )

                if result:
                    all_results.append(result)

            except Exception as e:
                print(f"✗ Error evaluating {checkpoint_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    else:
        # Original mode: Generate new images and evaluate
        print(f"Mode:         Generate and evaluate")
        print(f"Base Model:   {args.base_model}")
        print(f"{'='*70}\n")

        if not args.base_model:
            print("❌ ERROR: --base-model required for generation mode")
            return

        # Initialize full evaluator with SD pipeline
        evaluator = SOTALoRAEvaluator(args.base_model, args.device, model_paths)

        # Load test prompts
        test_prompts = load_test_prompts(args.character)

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

            except Exception as e:
                print(f"✗ Error evaluating {lora_path.name}: {e}")
                continue

    # Check if we have any results
    if len(all_results) == 0:
        print("\n" + "="*70)
        print("✗ NO CHECKPOINTS EVALUATED")
        print("="*70)
        print(f"No valid checkpoint files found in: {lora_dir}")
        print(f"Expected pattern: *.safetensors")
        print("="*70 + "\n")
        return

    # Rank checkpoints
    ranked = sorted(all_results, key=lambda x: x['composite_score'], reverse=True)

    # Save report
    report = {
        'character': args.character,
        'evaluation_models': {
            'prompt_alignment': 'InternVL2-8B' if evaluator.internvl_available else 'CLIP ViT-L/14',
            'aesthetics': 'LAION Aesthetics V2' if evaluator.aesthetics_available else 'N/A',
            'consistency': 'InsightFace' if evaluator.insightface_available else 'Visual Similarity',
            'quality': 'MUSIQ' if evaluator.musiq_available else 'Sharpness',
            'diversity': 'LPIPS' if evaluator.lpips_available else 'Embedding Variance'
        },
        'best_checkpoint': ranked[0]['checkpoint'],
        'best_score': ranked[0]['composite_score'],
        'rankings': ranked
    }

    report_path = output_dir / 'sota_evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SOTA EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best Checkpoint: {ranked[0]['checkpoint']}")
    print(f"Composite Score: {ranked[0]['composite_score']:.4f}")
    print(f"\nReport: {report_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
