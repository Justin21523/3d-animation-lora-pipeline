#!/usr/bin/env python3
"""
ComfyUI LoRA Testing Integration
Provides Python API to submit workflows to ComfyUI and retrieve results
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional

class ComfyUIClient:
    """Client for interacting with ComfyUI API"""

    def __init__(self, host: str = "localhost", port: int = 8188):
        self.base_url = f"http://{host}:{port}"
        self.client_id = "python_client"

    def submit_workflow(self, workflow: Dict) -> str:
        """Submit a workflow and return prompt_id"""
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id}
        )
        response.raise_for_status()
        return response.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> Optional[Dict]:
        """Get execution history for a prompt"""
        response = requests.get(f"{self.base_url}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()
        return history.get(prompt_id)

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict:
        """Wait for workflow completion and return results"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)

            if history and "outputs" in history:
                return history["outputs"]

            time.sleep(2)

        raise TimeoutError(f"Workflow {prompt_id} did not complete in {timeout}s")

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an image from ComfyUI"""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = requests.get(f"{self.base_url}/view", params=params)
        response.raise_for_status()
        return response.content


class LoRACheckpointTester:
    """Test multiple LoRA checkpoints using ComfyUI"""

    def __init__(self, workflow_path: Path, client: Optional[ComfyUIClient] = None):
        self.workflow_template = self._load_workflow(workflow_path)
        self.client = client or ComfyUIClient()

    @staticmethod
    def _load_workflow(path: Path) -> Dict:
        """Load workflow JSON template"""
        with open(path, 'r') as f:
            return json.load(f)

    def test_checkpoints(
        self,
        checkpoint_paths: List[Path],
        output_dir: Path,
        prompts: List[str],
        seeds: Optional[List[int]] = None
    ) -> Dict:
        """
        Test multiple LoRA checkpoints

        Args:
            checkpoint_paths: List of .safetensors files to test
            output_dir: Where to save results
            prompts: List of test prompts
            seeds: Optional list of seeds (defaults to [42])

        Returns:
            Dictionary mapping checkpoint names to output image paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = seeds or [42]

        results = {}

        for checkpoint in checkpoint_paths:
            print(f"Testing {checkpoint.name}...")

            # Modify workflow to use this checkpoint
            workflow = self._modify_workflow_for_checkpoint(checkpoint)

            # Submit workflow
            prompt_id = self.client.submit_workflow(workflow)

            # Wait for completion
            outputs = self.client.wait_for_completion(prompt_id)

            # Download results
            checkpoint_results = self._download_outputs(outputs, output_dir, checkpoint.stem)
            results[checkpoint.stem] = checkpoint_results

        return results

    def _modify_workflow_for_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Modify workflow template to use specific checkpoint"""
        workflow = self.workflow_template.copy()

        # Find LoRA loader node and update path
        # This depends on your workflow structure
        # Example: workflow["4"]["inputs"]["lora_name"] = checkpoint_path.name

        return workflow

    def _download_outputs(self, outputs: Dict, output_dir: Path, prefix: str) -> List[Path]:
        """Download all output images"""
        saved_images = []

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    filename = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")

                    # Download image
                    image_data = self.client.get_image(filename, subfolder)

                    # Save locally
                    output_path = output_dir / f"{prefix}_{filename}"
                    output_path.write_bytes(image_data)
                    saved_images.append(output_path)

        return saved_images


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LoRA checkpoints via ComfyUI")
    parser.add_argument("checkpoint_dir", type=Path, help="Directory containing .safetensors files")
    parser.add_argument("--workflow", type=Path,
                       default=Path("/mnt/c/ai_tools/comfyui/workflows/lora_testing/checkpoint_comparison.json"))
    parser.add_argument("--output-dir", type=Path,
                       default=Path("/mnt/data/outputs/comfyui/lora_tests"))
    parser.add_argument("--prompts", nargs="+",
                       default=["a 3d animated character, pixar style, neutral pose, studio lighting"])

    args = parser.parse_args()

    # Find all checkpoints
    checkpoints = list(args.checkpoint_dir.glob("*.safetensors"))

    # Run tests
    tester = LoRACheckpointTester(args.workflow)
    results = tester.test_checkpoints(checkpoints, args.output_dir, args.prompts)

    print(f"\nTesting complete! Results saved to {args.output_dir}")
    for name, images in results.items():
        print(f"  {name}: {len(images)} images")
