#!/bin/bash

# Build the image
docker build -t aider-lite .

# Run the container (example)
docker run -it --rm \
  --name aider-lite \
  -v $(pwd):/workspace \
  -w /workspace \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  aider-lite
