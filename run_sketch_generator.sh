#!/bin/bash

# Set the directory containing the images
IMAGE_DIR="./input"

# Iterate through all files in the directory
for image in "$IMAGE_DIR"/*
do
  # Check if it's a file
  if [ -f "$image" ]; then
    # Execute the sketch_generator.py script with the current file as an argument
    python sketch_generator.py "$image" --xdog --canny 
  fi
done

echo "Sketch generation completed for all images in $IMAGE_DIR"