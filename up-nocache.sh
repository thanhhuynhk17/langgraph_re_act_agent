#!/bin/bash
set -e

echo "🚀 Building without cache..."
docker compose build --no-cache
echo "🔼 Starting containers..."
docker compose up -d
echo "📜 Showing logs..."
docker compose logs -f
