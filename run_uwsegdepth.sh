#!/bin/bash

# arguments for extract_segms.py
SEG_MODEL="Swin"  # "Swin", "ConvNeXt"

# arguments for extract_depth_prior.py
INPUT_SPACE="RMI"  # "RMI", "RGB"

FRAME_INTERVAL=3

for file in ./video/*.mp4; do
    # arguments
    FILE_NAME=$(basename "$file" .mp4)
    VIDEO_PATH="$file"
    OUTPUT_FOLDER="./outputs/$FILE_NAME"
    FRAMES_FOLDER="$OUTPUT_FOLDER/frames"

    # Skip if already processed
    if [ -d "$OUTPUT_FOLDER" ]; then
        echo "Skipping $FILE_NAME: Output folder already exists at $OUTPUT_FOLDER"
        continue
    fi

    echo "=================================================="
    echo "| Starting Pipeline at  : $(date)"
    echo "| File Name             : $FILE_NAME"
    echo "| Video Path            : $VIDEO_PATH"
    echo "| Input Frame Interval  : $FRAME_INTERVAL"
    echo "| Frames Folder         : $FRAMES_FOLDER"
    echo "| Segmentation Model    : $SEG_MODEL"
    echo "| UDepth Input Space    : $INPUT_SPACE"
    echo "| Output FPS            : $FPS"
    echo "| Output Folder         : $OUTPUT_FOLDER"
    echo "=================================================="

    echo -e "\n =====> STEP (1 / 8): Video to Images \n"
    MAX_RETRIES=10
    RETRY_COUNT=0
    SUCCESS=false
    until [ "$SUCCESS" = true ] || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
        python video2img.py --video "$VIDEO_PATH" --frame_interval "$FRAME_INTERVAL" --output "$FRAMES_FOLDER"

        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            echo "❌ video2img.py failed (attempt $((RETRY_COUNT+1))/$MAX_RETRIES), retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 2
        fi
    done
    echo -e "\n ✅ Done: Extracted frames from video. \n"

    echo -e "\n =====> STEP (2 / 8): Generate CSV \n"
    MAX_RETRIES=10
    RETRY_COUNT=0
    SUCCESS=false
    until [ "$SUCCESS" = true ] || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
        python generate_csv.py --data_root "$FRAMES_FOLDER" --output_folder "$OUTPUT_FOLDER"

        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            echo "❌ generate_csv.py failed (attempt $((RETRY_COUNT+1))/$MAX_RETRIES), retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 2
        fi
    done
    echo -e "\n ✅ Done: Generated CSV file for each frames. \n"

    echo -e "\n =====> STEP (3 / 8): Instance Segmentation \n"
    MAX_RETRIES=10
    RETRY_COUNT=0
    SUCCESS=false
    until [ "$SUCCESS" = true ] || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
        python extract_segms.py --data_dir "$FRAMES_FOLDER" --output_dir "$OUTPUT_FOLDER" --model "$SEG_MODEL"

        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            echo "❌ extract_segms.py failed (attempt $((RETRY_COUNT+1))/$MAX_RETRIES), retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 2
        fi
    done
    echo -e "\n ✅ Done: Extracted instance segmentation masks. \n"

    echo -e "\n =====> STEP (4 / 8): Depth Prior (UDepth) \n"
    MAX_RETRIES=10
    RETRY_COUNT=0
    SUCCESS=false
    until [ "$SUCCESS" = true ] || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
        python extract_depth_prior.py --loc_folder "$FRAMES_FOLDER" --input_space "$INPUT_SPACE" --output_folder "$OUTPUT_FOLDER"

        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            echo "❌ extract_depth_prior.py failed (attempt $((RETRY_COUNT+1))/$MAX_RETRIES), retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 2
        fi
    done
    echo -e "\n ✅ Done: Generated UDepth-based depth prior maps. \n"

    echo -e "\n =====> STEP (5 / 8): Extract Sparse Features \n"
    MAX_RETRIES=10
    RETRY_COUNT=0
    SUCCESS=false
    until [ "$SUCCESS" = true ] || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
        python extract_features.py --index_file "$OUTPUT_FOLDER/inference.csv" --output "$OUTPUT_FOLDER"

        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            echo "❌ extract_features.py failed (attempt $((RETRY_COUNT+1))/$MAX_RETRIES), retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 2
        fi
    done
    echo -e "\n ✅ Done: Extracted sparse features from depth prior. \n"

    echo -e "\n =====> STEP (6 / 8): Depth Estimation (UWDepth + SADDER) \n"
    MAX_RETRIES=10
    RETRY_COUNT=0
    SUCCESS=false
    until [ "$SUCCESS" = true ] || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
        python extract_depth.py --index_file "$OUTPUT_FOLDER/inference.csv" --output_dir "$OUTPUT_FOLDER/vis_depth"

        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            echo "❌ extract_depth.py failed (attempt $((RETRY_COUNT+1))/$MAX_RETRIES), retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 2
        fi
    done
    echo -e "\n✅ Done: Estimated final depth maps with UWDepth + SADDER.\n"

    echo -e "\n =====> STEP (7 / 8): Depth-Informed Instance Segmentation \n"
    MAX_RETRIES=10
    RETRY_COUNT=0
    SUCCESS=false
    until [ "$SUCCESS" = true ] || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
        python extract_depth_segms.py --index_file "$OUTPUT_FOLDER/inference.csv" --output_dir "$OUTPUT_FOLDER/vis_depth_segms" --model "$SEG_MODEL"

        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            echo "❌ extract_depth_segms.py failed (attempt $((RETRY_COUNT+1))/$MAX_RETRIES), retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 2
        fi
    done
    echo -e "\n ✅ Done: Visualized segmentation + object-level depth. \n"

    echo -e "\n =====> STEP (8 / 8): Convert Images to Video \n"
    MAX_RETRIES=10
    RETRY_COUNT=0
    SUCCESS=false
    until [ "$SUCCESS" = true ] || [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; do
        python img2video.py --video "$VIDEO_PATH" --input "$FRAMES_FOLDER" --result "$OUTPUT_FOLDER/vis_depth_segms" --output "$OUTPUT_FOLDER" --frame_interval "$FRAME_INTERVAL"

        if [ $? -eq 0 ]; then
            SUCCESS=true
        else
            echo "❌ img2video.py failed (attempt $((RETRY_COUNT+1))/$MAX_RETRIES), retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 2
        fi
    done
    echo -e "\n 🎉 Done: Converted results back to video. \n"

    echo "===================== ALL STEPS COMPLETED SUCCESSFULLY ✅ ====================="
done
