#!/bin/bash
# Monitor caption generation progress

WORKSPACE="/mnt/data/ai_data/synthetic_lora_data/generated_data"

echo "========================================"
echo "Caption Generation Progress Monitor"
echo "========================================"
echo ""

TOTAL=0
for char in alberto alberto_seamonster barley_lightfoot bryce caleb elio giulia ian_lightfoot luca luca_seamonster miguel orion russell tyler; do
    echo "$char:"
    char_total=0
    for type in pose expression action; do
        count=$(find "$WORKSPACE/$char/$type/txt_captions" -name "*.txt" 2>/dev/null | wc -l)
        printf "  %-12s: %3d captions\n" "$type" "$count"
        char_total=$((char_total + count))
    done
    echo "  Total: $char_total"
    TOTAL=$((TOTAL + char_total))
    echo ""
done

echo "========================================"
echo "TOTAL CAPTIONS: $TOTAL / 4200 expected"
echo "Progress: $(echo "scale=1; $TOTAL * 100 / 4200" | bc)%"
echo "========================================"
