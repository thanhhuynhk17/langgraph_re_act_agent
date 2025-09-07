#!/bin/bash
set -e

echo "🛑 Stopping and removing containers, networks, volumes..."
docker compose down
echo "🚀 Building with cache..."
docker compose build
echo "🔼 Starting containers..."
docker compose up -d
echo "📜 Showing logs..."
docker compose logs -f
