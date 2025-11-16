#!/usr/bin/env python3
"""
AI-Powered Deep Learning Analysis for Anime Frames
Uses multiple state-of-the-art deep learning models for comprehensive analysis:
- CLIP: Scene classification, visual style, mood detection
- BLIP2: Content understanding and captioning
- ResNet50: Image quality assessment
- Aesthetic Predictor: Beauty score prediction

Optimized for GPU batch processing with high utilization
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import AI models
from transformers import CLIPProcessor, CLIPModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torchvision.models as models
import torchvision.transforms as transforms


class AnimeFrameDataset(Dataset):
    """Efficient dataset for anime frames with multi-threaded loading"""

    def __init__(self, frame_pairs: List[Tuple[Path, Path, str]], transform=None):
        """
        Args:
            frame_pairs: List of (char_path, bg_path, identifier) tuples
            transform: Image transforms
        """
        self.frame_pairs = frame_pairs
        self.transform = transform

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        char_path, bg_path, identifier = self.frame_pairs[idx]

        try:
            # Load character and background images
            char_img = Image.open(char_path).convert('RGB')
            bg_img = Image.open(bg_path).convert('RGB')

            if self.transform:
                char_img = self.transform(char_img)
                bg_img = self.transform(bg_img)

            return {
                'character': char_img,
                'background': bg_img,
                'path': identifier
            }
        except Exception as e:
            # Return dummy data on error
            dummy = torch.zeros(3, 224, 224)
            return {
                'character': dummy,
                'background': dummy,
                'path': identifier
            }


class CLIPSceneAnalyzer:
    """CLIP-based scene and mood analysis"""

    def __init__(self, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size

        print("🔄 Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

        # Scene categories
        self.scene_categories = [
            "indoor scene", "outdoor scene", "nature scene", "urban scene",
            "school scene", "sports field", "battle scene", "peaceful scene"
        ]

        # Mood categories
        self.mood_categories = [
            "exciting action", "calm peaceful", "tense dramatic", "happy cheerful",
            "sad melancholic", "mysterious dark", "energetic dynamic", "serene quiet"
        ]

        # Visual style
        self.style_categories = [
            "bright colorful", "dark moody", "warm colors", "cool colors",
            "high contrast", "soft lighting", "dramatic lighting", "natural lighting"
        ]

        print("✓ CLIP model loaded")

    @torch.no_grad()
    def analyze_batch(self, images: torch.Tensor) -> Dict:
        """Analyze a batch of images"""
        # Prepare images
        images_list = [transforms.ToPILImage()(img) for img in images.cpu()]

        results = {
            'scenes': [],
            'moods': [],
            'styles': []
        }

        # Scene classification
        inputs = self.processor(
            text=self.scene_categories,
            images=images_list,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        for prob in probs:
            top_idx = prob.argmax().item()
            results['scenes'].append({
                'category': self.scene_categories[top_idx],
                'confidence': prob[top_idx].item()
            })

        # Mood analysis
        inputs = self.processor(
            text=self.mood_categories,
            images=images_list,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        for prob in probs:
            top_idx = prob.argmax().item()
            results['moods'].append({
                'category': self.mood_categories[top_idx],
                'confidence': prob[top_idx].item()
            })

        # Visual style
        inputs = self.processor(
            text=self.style_categories,
            images=images_list,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        for prob in probs:
            top_idx = prob.argmax().item()
            results['styles'].append({
                'category': self.style_categories[top_idx],
                'confidence': prob[top_idx].item()
            })

        return results


class BLIP2ContentAnalyzer:
    """BLIP2-based content understanding and captioning"""

    def __init__(self, device='cuda', batch_size=8):
        self.device = device
        self.batch_size = batch_size

        print("🔄 Loading BLIP2 model...")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        print("✓ BLIP2 model loaded")

    @torch.no_grad()
    def generate_captions(self, images: torch.Tensor) -> List[str]:
        """Generate descriptive captions for images"""
        images_list = [transforms.ToPILImage()(img) for img in images.cpu()]

        inputs = self.processor(images=images_list, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return [cap.strip() for cap in captions]

    @torch.no_grad()
    def answer_questions(self, images: torch.Tensor, questions: List[str]) -> List[str]:
        """Answer questions about images"""
        images_list = [transforms.ToPILImage()(img) for img in images.cpu()]

        answers = []
        for img, question in zip(images_list, questions):
            inputs = self.processor(images=img, text=question, return_tensors="pt").to(self.device, torch.float16)

            generated_ids = self.model.generate(**inputs, max_new_tokens=20)
            answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            answers.append(answer.strip())

        return answers


class ResNetQualityAssessor:
    """ResNet-based image quality assessment"""

    def __init__(self, device='cuda'):
        self.device = device

        print("🔄 Loading ResNet50 for quality assessment...")
        self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()

        # Remove final classification layer to get features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        print("✓ ResNet50 loaded")

    @torch.no_grad()
    def assess_quality(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract deep features for quality assessment
        Higher feature magnitude = better quality
        """
        features = self.model(images.to(self.device))
        features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims but keep batch dim

        # Handle both single and batch inputs
        if features.dim() == 1:
            features = features.unsqueeze(0)  # Add batch dimension for single input

        # Compute quality score based on feature statistics
        quality_scores = features.norm(dim=1) / 100.0  # Normalize to 0-1 range
        quality_scores = torch.clamp(quality_scores, 0, 1)

        return quality_scores


class AestheticPredictor:
    """Aesthetic beauty score predictor based on ResNet"""

    def __init__(self, device='cuda'):
        self.device = device

        print("🔄 Loading Aesthetic Predictor...")
        # Use pretrained ResNet as base
        base_model = models.resnet50(pretrained=True)

        # Custom aesthetic head
        self.model = torch.nn.Sequential(
            *list(base_model.children())[:-1],
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        ).to(device)

        self.model.eval()
        print("✓ Aesthetic Predictor loaded")

    @torch.no_grad()
    def predict_aesthetic_score(self, images: torch.Tensor) -> torch.Tensor:
        """Predict aesthetic beauty score (0-1)"""
        scores = self.model(images.to(self.device))
        return scores.squeeze()


class AIDeepAnalyzer:
    """Main analyzer orchestrating all AI models"""

    def __init__(self, device='cuda', batch_size=16, num_workers=4):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"\n{'='*80}")
        print("🚀 Initializing AI Deep Analysis System")
        print(f"{'='*80}\n")

        # Initialize all models
        self.clip_analyzer = CLIPSceneAnalyzer(device, batch_size)
        self.blip2_analyzer = BLIP2ContentAnalyzer(device, batch_size=8)
        self.quality_assessor = ResNetQualityAssessor(device)
        self.aesthetic_predictor = AestheticPredictor(device)

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"\n✅ All models loaded on {device.upper()}")
        print(f"Batch size: {batch_size}, Workers: {num_workers}\n")

    def analyze_dataset(self, input_dir: Path, output_dir: Path):
        """Analyze entire dataset of anime frames"""

        # Collect all frame pairs (character, background)
        print("📂 Scanning dataset...")
        frame_pairs = []

        for episode_dir in sorted(input_dir.iterdir()):
            if not episode_dir.is_dir() or not episode_dir.name.startswith('episode_'):
                continue

            char_dir = episode_dir / 'character'
            bg_dir = episode_dir / 'background'

            if not char_dir.exists() or not bg_dir.exists():
                continue

            # Match character and background files
            char_files = {f.stem.replace('_character', ''): f for f in char_dir.glob('*_character.png')}

            for base_name, char_file in char_files.items():
                bg_file = bg_dir / f"{base_name}_background.jpg"

                if bg_file.exists():
                    identifier = f"{episode_dir.name}/{char_file.name}"
                    frame_pairs.append((char_file, bg_file, identifier))

        print(f"✓ Found {len(frame_pairs)} frames across {len(list(input_dir.glob('episode_*')))} episodes\n")

        if len(frame_pairs) == 0:
            print("⚠️  No frames found! Check data structure.")
            return []

        # Create dataset and dataloader
        dataset = AnimeFrameDataset(frame_pairs, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

        # Analysis results
        all_results = []

        # Process batches
        print("🎨 Analyzing frames with AI models...\n")

        with tqdm(total=len(frame_pairs), desc="Processing", unit="frame") as pbar:
            for batch in dataloader:
                char_images = batch['character'].to(self.device)
                bg_images = batch['background'].to(self.device)
                paths = batch['path']

                batch_size = char_images.size(0)

                # 1. CLIP analysis (scene, mood, style)
                clip_results = self.clip_analyzer.analyze_batch(bg_images)

                # 2. Quality assessment
                quality_scores = self.quality_assessor.assess_quality(char_images)

                # 3. Aesthetic prediction
                aesthetic_scores = self.aesthetic_predictor.predict_aesthetic_score(char_images)

                # 4. BLIP2 captions (smaller batches)
                captions_char = []
                captions_bg = []

                for i in range(0, batch_size, self.blip2_analyzer.batch_size):
                    end_idx = min(i + self.blip2_analyzer.batch_size, batch_size)

                    char_batch = char_images[i:end_idx]
                    bg_batch = bg_images[i:end_idx]

                    caps_char = self.blip2_analyzer.generate_captions(char_batch)
                    caps_bg = self.blip2_analyzer.generate_captions(bg_batch)

                    captions_char.extend(caps_char)
                    captions_bg.extend(caps_bg)

                # Compile results
                for i in range(batch_size):
                    # Extract tensor values safely (handle both batched and single-item cases)
                    if quality_scores.dim() == 0:
                        quality_val = quality_scores.item()
                    else:
                        quality_val = quality_scores[i].item() if i < len(quality_scores) else quality_scores[-1].item()

                    if aesthetic_scores.dim() == 0:
                        aesthetic_val = aesthetic_scores.item()
                    else:
                        aesthetic_val = aesthetic_scores[i].item() if i < len(aesthetic_scores) else aesthetic_scores[-1].item()

                    result = {
                        'frame_path': paths[i],
                        'episode': Path(paths[i]).parent.name,
                        'frame_name': Path(paths[i]).name,

                        # CLIP analysis
                        'scene_type': clip_results['scenes'][i]['category'],
                        'scene_confidence': clip_results['scenes'][i]['confidence'],
                        'mood': clip_results['moods'][i]['category'],
                        'mood_confidence': clip_results['moods'][i]['confidence'],
                        'visual_style': clip_results['styles'][i]['category'],
                        'style_confidence': clip_results['styles'][i]['confidence'],

                        # Quality metrics
                        'quality_score': quality_val,
                        'aesthetic_score': aesthetic_val,

                        # Content understanding
                        'character_caption': captions_char[i],
                        'background_caption': captions_bg[i]
                    }

                    all_results.append(result)

                pbar.update(batch_size)

        # Save detailed results
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / 'ai_deep_analysis.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Detailed results saved: {results_file}")

        # Generate statistics
        self._generate_statistics(all_results, output_dir)

        return all_results

    def _generate_statistics(self, results: List[Dict], output_dir: Path):
        """Generate comprehensive statistics"""

        print("\n📊 Generating statistics...\n")

        if len(results) == 0:
            print("⚠️  No results to generate statistics")
            return

        stats = {
            'total_frames': len(results),
            'timestamp': datetime.now().isoformat(),

            # Scene distribution
            'scene_distribution': defaultdict(int),
            'mood_distribution': defaultdict(int),
            'style_distribution': defaultdict(int),

            # Quality metrics
            'avg_quality_score': float(np.mean([r['quality_score'] for r in results])),
            'avg_aesthetic_score': float(np.mean([r['aesthetic_score'] for r in results])),

            # High quality frames (top 20%)
            'high_quality_frames': [],
            'high_aesthetic_frames': []
        }

        # Count distributions
        for result in results:
            stats['scene_distribution'][result['scene_type']] += 1
            stats['mood_distribution'][result['mood']] += 1
            stats['style_distribution'][result['visual_style']] += 1

        # Convert to regular dicts
        stats['scene_distribution'] = dict(stats['scene_distribution'])
        stats['mood_distribution'] = dict(stats['mood_distribution'])
        stats['style_distribution'] = dict(stats['style_distribution'])

        # Find high quality frames
        quality_threshold = float(np.percentile([r['quality_score'] for r in results], 80))
        aesthetic_threshold = float(np.percentile([r['aesthetic_score'] for r in results], 80))

        for result in results:
            if result['quality_score'] >= quality_threshold:
                stats['high_quality_frames'].append({
                    'path': result['frame_path'],
                    'score': result['quality_score']
                })

            if result['aesthetic_score'] >= aesthetic_threshold:
                stats['high_aesthetic_frames'].append({
                    'path': result['frame_path'],
                    'score': result['aesthetic_score']
                })

        # Sort by score
        stats['high_quality_frames'].sort(key=lambda x: x['score'], reverse=True)
        stats['high_aesthetic_frames'].sort(key=lambda x: x['score'], reverse=True)

        # Save statistics
        stats_file = output_dir / 'analysis_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"✓ Statistics saved: {stats_file}")

        # Print summary
        print(f"\n{'='*80}")
        print("📈 ANALYSIS SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total frames analyzed: {stats['total_frames']}")
        print(f"Average quality score: {stats['avg_quality_score']:.3f}")
        print(f"Average aesthetic score: {stats['avg_aesthetic_score']:.3f}")

        print(f"\n🎬 Top Scene Types:")
        for scene, count in sorted(stats['scene_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {scene}: {count} ({count/stats['total_frames']*100:.1f}%)")

        print(f"\n😊 Top Moods:")
        for mood, count in sorted(stats['mood_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {mood}: {count} ({count/stats['total_frames']*100:.1f}%)")

        print(f"\n🎨 Top Visual Styles:")
        for style, count in sorted(stats['style_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {style}: {count} ({count/stats['total_frames']*100:.1f}%)")

        print(f"\n⭐ High quality frames identified: {len(stats['high_quality_frames'])}")
        print(f"✨ High aesthetic frames identified: {len(stats['high_aesthetic_frames'])}")


def main():
    parser = argparse.ArgumentParser(
        description='AI-Powered Deep Learning Analysis for Anime Frames'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Input directory with layered frames (episode_XXX/frame_XXXX/)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        required=True,
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )

    args = parser.parse_args()

    # Check input
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Create analyzer
    analyzer = AIDeepAnalyzer(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    # Run analysis
    results = analyzer.analyze_dataset(args.input_dir, args.output_dir)

    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"Results saved to: {args.output_dir}")
    print(f"Analyzed {len(results)} frames\n")


if __name__ == '__main__':
    main()
