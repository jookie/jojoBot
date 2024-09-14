#!/bin/bash

# Create the root project directory
mkdir -p /jojobot

# Navigate to the project directory
cd /jojobot || exit

# Create the directory structure
mkdir -p api
mkdir -p app/api
mkdir -p app/train
mkdir -p app/
mkdir -p components
mkdir -p public
mkdir -p styles
mkdir -p scripts


# Create the necessary files
touch app/train/page.tsx
touch components/DataDisplay.tsx
touch package.json
touch next.config.js
touch next.config.js
touch .env

# Optional: Add a message to confirm completion
echo "Project structure created successfully!"
