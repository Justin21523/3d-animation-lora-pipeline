#!/usr/bin/env python3
"""
OpenAI API Cost Estimator for Yokai Watch Pipeline

Estimates API costs based on:
- Number of frames to annotate
- Model pricing (GPT-4o with vision)
- Sampling strategy
"""

# GPT-4o Pricing (as of 2024)
# https://openai.com/api/pricing/
GPT4O_INPUT_PRICE_PER_1M = 2.50  # $2.50 per 1M input tokens
GPT4O_OUTPUT_PRICE_PER_1M = 10.00  # $10.00 per 1M output tokens
GPT4O_IMAGE_PRICE_PER_IMAGE = 0.00255  # $0.00255 per image (1024x1024 detail=low)

# Alternative: GPT-4o-mini (cheaper)
GPT4O_MINI_INPUT_PRICE_PER_1M = 0.15
GPT4O_MINI_OUTPUT_PRICE_PER_1M = 0.60
GPT4O_MINI_IMAGE_PRICE_PER_IMAGE = 0.00255  # Same image cost

def estimate_costs(
    total_frames: int = 2_780_000,
    sample_rate: int = 50,  # 1 in 50 frames
    avg_input_tokens: int = 200,  # Prompt tokens
    avg_output_tokens: int = 150,  # Response tokens
    use_mini: bool = False
):
    """
    Estimate OpenAI API costs

    Args:
        total_frames: Total number of frames
        sample_rate: Sample 1 out of N frames
        avg_input_tokens: Average input tokens per request
        avg_output_tokens: Average output tokens per request
        use_mini: Use GPT-4o-mini instead of GPT-4o
    """
    print("="*80)
    print("OpenAI API Cost Estimation for Yokai Watch Pipeline")
    print("="*80)
    print()

    # Calculate frames to annotate
    frames_to_annotate = total_frames // sample_rate

    # Choose pricing
    if use_mini:
        model_name = "GPT-4o-mini"
        input_price = GPT4O_MINI_INPUT_PRICE_PER_1M
        output_price = GPT4O_MINI_OUTPUT_PRICE_PER_1M
        image_price = GPT4O_MINI_IMAGE_PRICE_PER_IMAGE
    else:
        model_name = "GPT-4o"
        input_price = GPT4O_INPUT_PRICE_PER_1M
        output_price = GPT4O_OUTPUT_PRICE_PER_1M
        image_price = GPT4O_IMAGE_PRICE_PER_IMAGE

    print(f"Model: {model_name}")
    print(f"Total Frames: {total_frames:,}")
    print(f"Sample Rate: 1 in {sample_rate}")
    print(f"Frames to Annotate: {frames_to_annotate:,}")
    print()

    # Cost breakdown
    print("Cost Breakdown:")
    print("-" * 80)

    # Image processing cost
    image_cost = frames_to_annotate * image_price
    print(f"Image Processing: {frames_to_annotate:,} images × ${image_price} = ${image_cost:.2f}")

    # Input tokens cost
    total_input_tokens = frames_to_annotate * avg_input_tokens
    input_cost = (total_input_tokens / 1_000_000) * input_price
    print(f"Input Tokens: {total_input_tokens:,} tokens × ${input_price}/1M = ${input_cost:.2f}")

    # Output tokens cost
    total_output_tokens = frames_to_annotate * avg_output_tokens
    output_cost = (total_output_tokens / 1_000_000) * output_price
    print(f"Output Tokens: {total_output_tokens:,} tokens × ${output_price}/1M = ${output_cost:.2f}")

    # Total
    total_cost = image_cost + input_cost + output_cost
    print("-" * 80)
    print(f"TOTAL ESTIMATED COST: ${total_cost:.2f}")
    print()

    return total_cost

def show_recommendations():
    """Show cost-saving recommendations"""
    print()
    print("="*80)
    print("Cost Optimization Recommendations")
    print("="*80)
    print()

    print("1. Smart Sampling Strategies:")
    print("   - Sample rate 1:50 → Annotate ~55,600 frames → ~$142-$570")
    print("   - Sample rate 1:100 → Annotate ~27,800 frames → ~$71-$285")
    print("   - Sample rate 1:200 → Annotate ~13,900 frames → ~$36-$143")
    print()

    print("2. Priority-Based Annotation:")
    print("   - Annotate ALL summoning scenes (high value)")
    print("   - Sample normal scenes at lower rate")
    print("   - Skip repetitive/similar frames")
    print()

    print("3. Model Selection:")
    print("   - GPT-4o: Better quality, $142-$570 for 1:50 sampling")
    print("   - GPT-4o-mini: 85% quality, $22-$90 for 1:50 sampling (87% cheaper!)")
    print("   - Hybrid: Use GPT-4o for key frames, mini for others")
    print()

    print("4. Batch Processing:")
    print("   - Process in batches to monitor spending")
    print("   - Set budget limits ($50, $100, $200)")
    print("   - Stop if approaching limit")
    print()

if __name__ == "__main__":
    print()
    print("Scenario 1: Conservative (Sample 1:50, GPT-4o)")
    estimate_costs(sample_rate=50, use_mini=False)

    print()
    print("Scenario 2: Economical (Sample 1:50, GPT-4o-mini)")
    estimate_costs(sample_rate=50, use_mini=True)

    print()
    print("Scenario 3: Balanced (Sample 1:100, GPT-4o-mini)")
    estimate_costs(sample_rate=100, use_mini=True)

    print()
    print("Scenario 4: Ultra-light (Sample 1:200, GPT-4o-mini)")
    estimate_costs(sample_rate=200, use_mini=True)

    show_recommendations()

    print()
    print("="*80)
    print("Recommendation: Start with Scenario 3 (Sample 1:100, GPT-4o-mini)")
    print("Estimated Cost: ~$11-45")
    print("="*80)
    print()
