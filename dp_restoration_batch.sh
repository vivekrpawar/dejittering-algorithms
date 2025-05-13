#!/bin/bash 

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_folder> <output_file> <max_shift> <alpha>"
    exit 1
fi

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
 
mkdir -p "$OUTPUT_FOLDER"
 
for video_file in "$INPUT_FOLDER"/*.{mp4,avi}; do   
    [ -e "$video_file" ] || continue 

    # video_name=$(basename "$video_file" .mp4)
    video_name=$(basename "$video_file" .avi)  
    output_path="$OUTPUT_FOLDER/$video_name"
    python dejittering_with_dp_sequence_dsf_acc2_video.py "$video_file" "$output_path" $3 $4
    
    echo "Processed: $video_file -> $output_path"
done