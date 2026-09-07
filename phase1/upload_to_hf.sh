#!/bin/bash
echo "Waiting for the compression process to finish..."
while pgrep -x tar > /dev/null; do
    sleep 5
done
echo "Compression finished! Starting HuggingFace upload..."
python3 /home/manya/LMA/upload_to_hf.py > /home/manya/LMA/upload_to_hf.log 2>&1
echo "Upload complete! Check upload_to_hf.log for details."
