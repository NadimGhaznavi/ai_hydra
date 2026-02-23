#!/bin/bash
#

# Exsit on error
set -e

# Clear the terminal
clear

# Project name
AI_HYDRA="ai_hydra"

# Source the functions file
FUNCTIONS="hydra-release-functions.sh"
SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd -- "$SCRIPTS_DIR/.." && pwd)"

if [ -e "$SCRIPTS_DIR/$FUNCTIONS" ]; then
	source "$SCRIPTS_DIR/$FUNCTIONS"
else
	echo "FATAL ERROR: Unable to find functions file: $SCRIPTS_DIR/$FUNCTIONS"
	exit 1
fi

cd $BASE_DIR

echo "🔍 Executing pre-release tests..."
echo $DIV

echo "📝 Running flake8..."
flake8 $AI_HYDRA
echo $DIV

echo "🔍 Running mypy..."
mypy $AI_HYDRA
echo $DIV

echo "🎨 Running black ..."
black $AI_HYDRA
echo $DIV

echo "📦 Running isort ..."
isort $AI_HYDRA

echo "🔒 Running bandit security check..."
bandit -r $AI_HYDRA #--skip B101

echo "🧹 Executing: poetry run pytest..."
poetry run pytest
echo $DIV

echo "🚦 Executing: shrmt -w scripts/..."
shfmt -w scripts/
echo $DIV

echo "👽 Executging: poetry run pre-commit run --all-files ..."
poetry run pre-commit run --all-files
echo $DIV

echo "🗃️ Rebuilding documentation ..."
cd $BASE_DIR/docs && make clean
cd $BASE_DIR/docs && make html
echo $DIV

echo "✅ All code quality checks passed!"
