#!/usr/bin/env bash
# Builds the NeuronSteer demo video from text slides with ffmpeg drawtext.
# Silent, 1280x720, ~74 seconds. No downloads, no external assets.
# Text goes through textfile= so commas/colons in copy cannot break parsing.
set -euo pipefail
cd "$(dirname "$0")"
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BG="0x0d1117"
mkdir -p frames txt

slide() {
  # name duration title subtitle accent
  local name="$1" dur="$2" title="$3" sub="$4" accent="${5:-0x3fb950}"
  printf '%s' "$title" > "txt/${name}_t.txt"
  printf '%s' "$sub" > "txt/${name}_s.txt"
  ffmpeg -y -v error -f lavfi -i "color=c=${BG}:s=1280x720:r=30:d=${dur}" \
    -vf "drawbox=x=0:y=0:w=1280:h=6:color=${accent}:t=fill,\
drawtext=fontfile=${FONT}:textfile=txt/${name}_t.txt:fontcolor=white:fontsize=52:x=(w-text_w)/2:y=250,\
drawtext=fontfile=${FONT_REG}:textfile=txt/${name}_s.txt:fontcolor=0x9da7b3:fontsize=31:x=(w-text_w)/2:y=350" \
    -c:v libx264 -pix_fmt yuv420p "frames/${name}.mp4"
}

slide 01 8  "NeuronSteer" "Neuron steering for Hermes Agent"
slide 02 10 "Nous Research, productised" "Their method: Contrastive Neuron Attribution, arXiv 2605.12290"
slide 03 12 "1. FIND the circuit" "Two prompt sets. Forward passes only. No training, no weight changes."
slide 04 12 "2. STEER with it" "Scale the neurons: 0 ablates, 1 is baseline, above 1 amplifies."
slide 05 12 "3. SCAN the dose window" "Behavior rate and output quality at every multiplier."
slide 06 10 "Four tools for the agent" "find, generate, scan, list, plus a CLI and a slash command."
slide 07 10 "13 of 13 tests passing" "github.com/Mrfentmen/hermes-neuron-steer"

printf "file '%s'\n" 01.mp4 02.mp4 03.mp4 04.mp4 05.mp4 06.mp4 07.mp4 \
  > frames/list.txt
ffmpeg -y -v error -f concat -safe 0 -i frames/list.txt -c copy neuronsteer-demo.mp4
ls -la neuronsteer-demo.mp4
