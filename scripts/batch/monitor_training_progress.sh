#!/bin/bash
#
# Real-time Training Progress Monitor
# Monitors GPU usage, checkpoints, and tensorboard metrics
#

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default values
OUTPUT_DIR="${1:-/mnt/data/ai_data/models/lora_sdxl/coco/miguel_identity}"
REFRESH_INTERVAL=5

clear
echo "================================================================================"
echo "🔍 SDXL LoRA Training Progress Monitor"
echo "================================================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Refresh interval: ${REFRESH_INTERVAL}s"
echo "Press Ctrl+C to exit"
echo "================================================================================"
echo ""

# Get initial checkpoint count
LAST_CHECKPOINT_COUNT=0

while true; do
    # Clear screen for refresh
    tput cup 7 0
    tput ed
    
    # Show timestamp
    echo -e "${BLUE}=== $(date '+%Y-%m-%d %H:%M:%S') ===${NC}"
    echo ""
    
    # GPU Status
    echo -e "${GREEN}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        awk -F, '{printf "  GPU %s: %s%% | VRAM: %s/%s MB | Temp: %s°C | Power: %sW\n", $1, $3, $4, $5, $6, $7}'
    echo ""
    
    # Training Process
    echo -e "${GREEN}Training Process:${NC}"
    TRAIN_PID=$(ps aux | grep "sdxl_train_network.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$TRAIN_PID" ]; then
        RUNTIME=$(ps -p "$TRAIN_PID" -o etime= | tr -d ' ')
        CPU_USAGE=$(ps -p "$TRAIN_PID" -o %cpu= | tr -d ' ')
        MEM_USAGE=$(ps -p "$TRAIN_PID" -o %mem= | tr -d ' ')
        echo -e "  PID: $TRAIN_PID | Runtime: $RUNTIME | CPU: ${CPU_USAGE}% | MEM: ${MEM_USAGE}%"
    else
        echo -e "  ${RED}No training process found${NC}"
    fi
    echo ""
    
    # Checkpoints
    echo -e "${GREEN}Checkpoints:${NC}"
    if [ -d "$OUTPUT_DIR" ]; then
        CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.safetensors" 2>/dev/null | wc -l)
        if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
            echo -e "  Total checkpoints: ${CHECKPOINT_COUNT}"
            echo -e "  Latest checkpoints:"
            find "$OUTPUT_DIR" -maxdepth 1 -name "*.safetensors" -type f -printf "    %TY-%Tm-%Td %TH:%TM  %f\n" 2>/dev/null | sort -r | head -5
            
            # Check for new checkpoint
            if [ "$CHECKPOINT_COUNT" -gt "$LAST_CHECKPOINT_COUNT" ]; then
                echo -e "  ${YELLOW}🔔 New checkpoint detected!${NC}"
                LAST_CHECKPOINT_COUNT=$CHECKPOINT_COUNT
            fi
        else
            echo "  No checkpoints yet"
        fi
    else
        echo -e "  ${RED}Output directory not found${NC}"
    fi
    echo ""
    
    # TensorBoard Logs
    echo -e "${GREEN}TensorBoard Metrics:${NC}"
    LATEST_LOG_DIR=$(find "$OUTPUT_DIR/logs" -name "network_train" -type d 2>/dev/null | sort -r | head -1)
    if [ -n "$LATEST_LOG_DIR" ]; then
        LATEST_TFEVENTS=$(find "$LATEST_LOG_DIR" -name "events.out.tfevents.*" -type f 2>/dev/null | sort -r | head -1)
        if [ -n "$LATEST_TFEVENTS" ]; then
            LOG_SIZE=$(du -h "$LATEST_TFEVENTS" | cut -f1)
            LOG_TIME=$(stat -c %y "$LATEST_TFEVENTS" | cut -d'.' -f1)
            echo "  Latest log: $(basename $LATEST_TFEVENTS)"
            echo "  Size: $LOG_SIZE | Last updated: $LOG_TIME"
            echo "  View in TensorBoard: tensorboard --logdir='$OUTPUT_DIR/logs' --port=6006"
        fi
    else
        echo "  No tensorboard logs found"
    fi
    echo ""
    
    # Dataset info (only show once)
    if [ "$LAST_CHECKPOINT_COUNT" -eq 0 ]; then
        echo -e "${GREEN}Training Configuration:${NC}"
        CONFIG_FILE=$(ps aux | grep "sdxl_train_network.py" | grep -v grep | grep -o "\--config_file=[^ ]*" | cut -d'=' -f2)
        if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
            EPOCHS=$(grep "max_train_epochs" "$CONFIG_FILE" | grep -o "[0-9]*" | head -1)
            SAVE_INTERVAL=$(grep "save_every_n_epochs" "$CONFIG_FILE" | grep -o "[0-9]*" | head -1)
            BATCH_SIZE=$(grep "train_batch_size" "$CONFIG_FILE" | grep -o "[0-9]*" | head -1)
            echo "  Config: $(basename $CONFIG_FILE)"
            echo "  Epochs: $EPOCHS | Save interval: every $SAVE_INTERVAL epochs | Batch size: $BATCH_SIZE"
        fi
        echo ""
    fi
    
    echo "================================================================================"
    echo "Next update in ${REFRESH_INTERVAL}s... (Ctrl+C to exit)"
    
    sleep "$REFRESH_INTERVAL"
done
