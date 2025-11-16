#!/usr/bin/env python3
"""
使用 VLM (Qwen2-VL / InternVL2) 全面重新生成 captions

根據 SOTA 分析結果，生成優化的 captions，包含：
1. Pixar 統一光照描述
2. 詳細角色特徵
3. 低對比度強調
4. 電影級色彩調性
5. 3D 材質描述

支持的 VLM：
- Qwen2-VL (推薦)
- InternVL2
- BLIP2 (備選)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class VLMCaptionGenerator:
    """VLM Caption 生成器"""

    def __init__(
        self,
        model_name: str = "qwen2_vl",
        device: str = "cuda",
        character_profile: Dict = None
    ):
        self.model_name = model_name
        self.device = device
        self.character_profile = character_profile or {}
        self.model = None
        self.processor = None

        print(f"🚀 初始化 {model_name} 模型...")
        self.load_model()

    def load_model(self):
        """加載 VLM 模型"""

        if self.model_name == "qwen2_vl":
            self._load_qwen2_vl()
        elif self.model_name == "internvl2":
            self._load_internvl2()
        elif self.model_name == "blip2":
            self._load_blip2()
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

    def _load_qwen2_vl(self):
        """加載 Qwen2-VL"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            # 使用本地 ai_warehouse 中的模型
            model_id = "/mnt/c/AI_LLM_projects/ai_warehouse/models/vlm/Qwen2-VL-7B-Instruct"
            print(f"  📦 加載 Qwen2-VL (本地): {model_id}")

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_id)

            print("    ✓ Qwen2-VL 已加載")

        except Exception as e:
            print(f"    ⚠️  Qwen2-VL 加載失敗: {e}")
            print("    💡 安裝: pip install transformers>=4.37.0")
            raise

    def _load_internvl2(self):
        """加載 InternVL2"""
        try:
            from transformers import AutoModel, AutoTokenizer

            model_id = "OpenGVLab/InternVL2-8B"
            print(f"  📦 加載 InternVL2: {model_id}")

            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            print("    ✓ InternVL2 已加載")

        except Exception as e:
            print(f"    ⚠️  InternVL2 加載失敗: {e}")
            raise

    def _load_blip2(self):
        """加載 BLIP2（備選）"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration

            model_id = "Salesforce/blip2-opt-2.7b"
            print(f"  📦 加載 BLIP2: {model_id}")

            self.processor = Blip2Processor.from_pretrained(model_id)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            print("    ✓ BLIP2 已加載")

        except Exception as e:
            print(f"    ⚠️  BLIP2 加載失敗: {e}")
            raise

    def create_prompt(self, analysis_result: Optional[Dict] = None) -> str:
        """
        創建 VLM prompt

        根據分析結果調整 prompt 以修正問題
        """

        # 基礎 prompt 模板（詳細指導，確保完整描述）
        base_prompt = """Analyze this 3D animated character image from Pixar's animation style and generate a detailed caption.

CRITICAL REQUIREMENTS (PRIORITIZED BY IMPORTANCE):
1. FACIAL FEATURES (HIGHEST PRIORITY): Provide EXTREMELY DETAILED facial descriptions:
   - Face shape, eyes (size, color, shape), eyebrows (thickness, arch), nose (size, shape)
   - Mouth expression, skin tone and texture, age-related features

2. EXPRESSION DETAILS: Describe the character's emotional state and facial expression:
   - Emotional state, eye expression, eyebrow movement, mouth expression

3. POSE & ACTION: Describe body position, gesture, and movement:
   - Body pose, head position, arm/hand gesture, specific action, camera angle/view

4. STYLE & RENDERING: "pixar film quality, 3d animation, cinematic rendering, smooth shading, subsurface scattering (SSS)"

5. LIGHTING: "pixar uniform lighting, even illumination, low contrast, subtle ambient lighting"

6. MATERIALS: "physically based rendering (PBR), matte skin shader, realistic fabric materials"

7. COLOR: "film color grading, balanced saturation, warm/cool tones"

"""

        # 如果有角色 profile，添加角色特定信息（MUST BE INCLUDED IN OUTPUT）
        if self.character_profile:
            base_prompt += f"\n🔴 MANDATORY CHARACTER IDENTITY (MUST APPEAR IN CAPTION):\n"
            base_prompt += "The character in this image is:\n"
            if 'core_description' in self.character_profile:
                base_prompt += f"⚠️ YOU MUST INCLUDE THIS EXACT DESCRIPTION IN YOUR CAPTION:\n"
                base_prompt += f'"{self.character_profile["core_description"]}"\n\n'
            if 'name' in self.character_profile:
                base_prompt += f"- Character name: {self.character_profile['name']}\n"
            if 'full_name' in self.character_profile:
                base_prompt += f"- Full name: {self.character_profile['full_name']}\n"
            if 'film' in self.character_profile:
                base_prompt += f"- From movie: {self.character_profile['film']}\n"
            if 'age' in self.character_profile:
                base_prompt += f"- Age: {self.character_profile['age']}\n"
            if 'physical_traits' in self.character_profile:
                base_prompt += f"- Key physical traits: {self.character_profile['physical_traits']}\n"

        base_prompt += """
TASK: Complete the caption template below by filling in [BLANKS] based on what you see in the image.

FIXED TEMPLATE (FILL IN THE BLANKS):
"a 3d animated character, pixar uniform lighting, even illumination, Luca Paguro from Pixar Luca (2021), 12-year-old italian pre-teen boy, large round brown eyes, thick arched eyebrows, button red-tinted nose, rosy cheeks, soft oval face, short dark-brown wavy curls, [HAIR_DETAILS], [EXPRESSION], [POSE_ACTION], pixar film quality, smooth shading, subsurface scattering on skin, matte skin shader, [CLOTHING_DETAILS], cinematic rendering, physically based rendering."

FILL IN THESE BLANKS:
1. [HAIR_DETAILS]: Describe hair style/state (e.g., "front quiff visible", "curls slightly tousled", "neat side part")
2. [EXPRESSION]: Facial expression (e.g., "surprised expression with wide eyes and raised eyebrows", "worried look with furrowed brows", "happy smile")
3. [POSE_ACTION]: Body pose and action (e.g., "lying on stomach underwater", "standing barefoot against off-white background", "looking up with open mouth")
4. [CLOTHING_DETAILS]: What is he wearing? (e.g., "barefoot with green mermaid tail", "barefoot in casual clothing", "shirtless")

EXAMPLE 1 (63 words):
"a 3d animated character, pixar uniform lighting, even illumination, Luca Paguro from Pixar Luca (2021), 12-year-old italian pre-teen boy, large round brown eyes, thick arched eyebrows, button red-tinted nose, rosy cheeks, soft oval face, short dark-brown wavy curls, front quiff visible, surprised expression with wide eyes, lying on stomach underwater, pixar film quality, smooth shading, subsurface scattering on skin, matte skin shader, green mermaid tail, cinematic rendering, physically based rendering."

EXAMPLE 2 (61 words):
"a 3d animated character, pixar uniform lighting, even illumination, Luca Paguro from Pixar Luca (2021), 12-year-old italian pre-teen boy, large round brown eyes, thick arched eyebrows, button red-tinted nose, rosy cheeks, soft oval face, short dark-brown wavy curls, neat side part, worried look with furrowed brows, standing barefoot against off-white background, pixar film quality, smooth shading, subsurface scattering on skin, matte skin shader, barefoot in casual wear, cinematic rendering, physically based rendering."

RULES:
- Output ONLY the complete filled template (one sentence, ending with period)
- Keep each blank SHORT (3-8 words)
- Total: 60-75 words target (important info in first 50 words)
- DO NOT add extra sentences or information
- DO NOT repeat any part of the template
"""

        return base_prompt

    def generate_caption(
        self,
        image: Image.Image,
        analysis_result: Optional[Dict] = None
    ) -> str:
        """生成單張圖像的 caption"""

        prompt = self.create_prompt(analysis_result)

        if self.model_name == "qwen2_vl":
            return self._generate_qwen2_vl(image, prompt)
        elif self.model_name == "internvl2":
            return self._generate_internvl2(image, prompt)
        elif self.model_name == "blip2":
            return self._generate_blip2(image, prompt)

    def _generate_qwen2_vl(self, image: Image.Image, prompt: str) -> str:
        """使用 Qwen2-VL 生成"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        # 記錄輸入長度，以便只解碼新生成的 tokens
        input_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,  # 增加到 120 tokens (約 80-90 words) - 包含完整風格/材質描述
                do_sample=False,
                min_new_tokens=80,   # 最少 80 tokens (約 60 words) - 確保包含核心資訊
                repetition_penalty=1.3  # 防止重複段落（1.0=無懲罰, 1.3=適度懲罰重複）
            )

        # 只解碼新生成的 tokens（不包括輸入 prompt）
        generated_tokens = outputs[:, input_length:]
        caption = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        return self._clean_caption(caption)

    def _generate_internvl2(self, image: Image.Image, prompt: str) -> str:
        """使用 InternVL2 生成"""

        pixel_values = self.model.vision_model.preprocess(image).to(
            self.device,
            dtype=torch.float16
        )

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config={
                    'max_new_tokens': 150,
                    'do_sample': False
                }
            )

        return self._clean_caption(response)

    def _generate_blip2(self, image: Image.Image, prompt: str) -> str:
        """使用 BLIP2 生成"""

        inputs = self.processor(image, text=prompt, return_tensors="pt").to(
            self.device,
            torch.float16
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                min_new_tokens=50
            )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)

        return self._clean_caption(caption)

    def _clean_caption(self, caption: str) -> str:
        """清理和規範化 caption"""
        import re

        # 移除可能的對話標記
        caption = caption.replace("Assistant:", "").replace("AI:", "")
        caption = caption.strip()

        # 移除重複的前綴（VLM 有時會重複 "a 3d animated character"）
        prefix_pattern = r'^["\']?\s*a 3d animated character,?\s*'
        # 移除所有前綴出現
        while re.search(prefix_pattern, caption, flags=re.IGNORECASE):
            caption = re.sub(prefix_pattern, '', caption, count=1, flags=re.IGNORECASE).strip()

        # 移除引號
        caption = caption.replace('"', '').replace("'", '')

        # 添加統一前綴（確保一致性）
        caption = 'a 3d animated character, ' + caption

        # 確保結尾有句號
        if caption and caption[-1] not in '.!?':
            caption += '.'

        return caption

    def batch_generate(
        self,
        image_dir: Path,
        output_dir: Path,
        analysis_result: Optional[Dict] = None,
        sample_size: Optional[int] = None
    ) -> Dict:
        """批量生成 captions"""

        print(f"\n📸 開始批量生成 captions...")
        print(f"  輸入目錄: {image_dir}")
        print(f"  輸出目錄: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 獲取所有圖像
        image_files = sorted(list(image_dir.glob("*.png")))

        if sample_size:
            image_files = image_files[:sample_size]

        print(f"  找到 {len(image_files)} 張圖像")

        results = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'captions': []
        }

        skipped = 0
        for img_path in tqdm(image_files, desc="  生成中"):
            try:
                # 檢查是否已有 caption（跳過已存在的）
                txt_path = output_dir / f"{img_path.stem}.txt"
                if txt_path.exists():
                    skipped += 1
                    continue

                # 加載圖像
                image = Image.open(img_path).convert('RGB')

                # 生成 caption
                caption = self.generate_caption(image, analysis_result)

                # 保存 caption
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)

                results['success'] += 1
                results['captions'].append({
                    'image': img_path.name,
                    'caption': caption
                })

            except Exception as e:
                print(f"\n  ⚠️  處理失敗: {img_path.name} - {e}")
                results['failed'] += 1

        print(f"\n✅ 完成！")
        print(f"  成功: {results['success']}")
        print(f"  跳過: {skipped} (已有 caption)")
        print(f"  失敗: {results['failed']}")

        # 保存元數據
        metadata_path = output_dir / "caption_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results


def load_character_profile(profile_path: Path) -> Dict:
    """加載角色 profile"""
    if profile_path and profile_path.exists():
        import yaml
        with open(profile_path, 'r', encoding='utf-8') as f:
            if profile_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    return {}


def load_analysis_result(analysis_path: Path) -> Optional[Dict]:
    """加載 SOTA 分析結果"""
    if analysis_path and analysis_path.exists():
        with open(analysis_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="使用 VLM 全面重新生成優化的 captions"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="圖像目錄"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="輸出目錄（預設：覆蓋原始 caption）"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen2_vl", "internvl2", "blip2"],
        default="qwen2_vl",
        help="VLM 模型選擇"
    )
    parser.add_argument(
        "--character-profile",
        type=Path,
        default=None,
        help="角色 profile JSON 文件"
    )
    parser.add_argument(
        "--analysis-result",
        type=Path,
        default=None,
        help="SOTA 分析結果 JSON 文件（用於針對性優化）"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="僅處理前 N 張圖像（測試用）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="設備 (cuda/cpu)"
    )

    args = parser.parse_args()

    if not args.image_dir.exists():
        print(f"錯誤：圖像目錄不存在: {args.image_dir}")
        return 1

    # 輸出目錄
    output_dir = args.output_dir or args.image_dir

    # 加載角色 profile
    character_profile = load_character_profile(args.character_profile)

    # 加載分析結果
    analysis_result = load_analysis_result(args.analysis_result)

    if analysis_result:
        print(f"✓ 已加載 SOTA 分析結果，將針對性優化 captions")

    # 創建生成器
    generator = VLMCaptionGenerator(
        model_name=args.model,
        device=args.device,
        character_profile=character_profile
    )

    # 批量生成
    results = generator.batch_generate(
        image_dir=args.image_dir,
        output_dir=output_dir,
        analysis_result=analysis_result,
        sample_size=args.sample_size
    )

    # 顯示樣本
    print(f"\n📝 Caption 樣本（前 3 個）：")
    for i, item in enumerate(results['captions'][:3], 1):
        print(f"\n{i}. {item['image']}")
        print(f"   {item['caption'][:100]}...")

    return 0


if __name__ == "__main__":
    exit(main())
