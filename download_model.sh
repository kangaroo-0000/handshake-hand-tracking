#!/bin/bash

# Check if wget or curl is installed
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the file."
    exit 1
fi

# Define the URL for the specific .pt file
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
TARGET_FILE="sam2.1_hiera_large.pt"
FILE_URL="${BASE_URL}/${TARGET_FILE}"

# Download the .pt file
echo "Downloading ${TARGET_FILE}..."
$CMD $FILE_URL || { echo "Failed to download the file from ${FILE_URL}"; exit 1; }

echo "Download completed successfully."
