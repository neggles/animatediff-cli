#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(git rev-parse --show-toplevel)
if [[ ! -d "${repo_dir}" ]]; then
    echo "Could not find the repo root. Checking for ./data/models/sd"
    repo_dir="."
fi

models_dir=$(realpath "${repo_dir}/data/models/sd")
if [[ ! -d "${models_dir}" ]]; then
    echo "Could not find repo root or models directory."
    echo "Either create ./data/models/sd or run this script from a checked-out git repo."
    exit 1
fi

model_urls=(
    https://civitai.com/api/download/models/78775 # ToonYou
    https://civitai.com/api/download/models/72396 # Lyriel
    https://civitai.com/api/download/models/71009 # RcnzCartoon
    https://civitai.com/api/download/models/79068 # MajicMix
    https://civitai.com/api/download/models/29460 # RealisticVision
    https://civitai.com/api/download/models/97261 # Tusun (1/2)
    https://civitai.com/api/download/models/50705 # Tusun (2/2)
    https://civitai.com/api/download/models/90115 # FilmVelvia (1/2)
    https://civitai.com/api/download/models/92475 # FilmVelvia (2/2)
    https://civitai.com/api/download/models/102828 # GhibliBackground (1/2)
    https://civitai.com/api/download/models/57618 # GhibliBackground (2/2)
)

echo "Downloading model files to ${models_dir}..."

# Create the models directory if it doesn't exist
mkdir -p "${models_dir}"

# Download the models
for url in ${model_urls[@]}; do
    curl -JLO --output-dir "${models_dir}" "${url}" || true
done
