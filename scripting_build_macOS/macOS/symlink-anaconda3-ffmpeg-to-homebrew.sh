#!/usr/bin/env bash

set -e

echo "----------------------------------------"
echo "Configure cvvdp env to use Homebrew FFmpeg"
echo "----------------------------------------"

echo
echo "Detecting Homebrew..."

BREW=$(command -v brew)

if [ -z "$BREW" ]; then
    echo "ERROR: Homebrew not found."
    exit 1
fi

BREW_PREFIX=$($BREW --prefix)

FFMPEG_SRC="$BREW_PREFIX/bin/ffmpeg"
FFPROBE_SRC="$BREW_PREFIX/bin/ffprobe"

echo "Homebrew prefix: $BREW_PREFIX"

if [ ! -f "$FFMPEG_SRC" ]; then
    echo "ERROR: ffmpeg not found in Homebrew:"
    echo "$FFMPEG_SRC"
    exit 1
fi

if [ ! -f "$FFPROBE_SRC" ]; then
    echo "ERROR: ffprobe not found in Homebrew."
    exit 1
fi

echo
echo "Locating cvvdp conda environment..."

ENV_BIN=""

for root in \
"$BREW_PREFIX/anaconda3/envs" \
"$BREW_PREFIX/Caskroom/miniconda/base/envs" \
"$HOME/miniconda3/envs" \
"$HOME/anaconda3/envs"
do
    if [ -d "$root/cvvdp/bin" ]; then
        ENV_BIN="$root/cvvdp/bin"
        break
    fi
done

if [ -z "$ENV_BIN" ]; then
    echo "ERROR: Could not find cvvdp environment."
    exit 1
fi

echo "cvvdp env bin: $ENV_BIN"

FFMPEG_DST="$ENV_BIN/ffmpeg"
FFPROBE_DST="$ENV_BIN/ffprobe"

echo
echo "Removing existing binaries..."

[ -e "$FFMPEG_DST" ] && rm "$FFMPEG_DST"
[ -e "$FFPROBE_DST" ] && rm "$FFPROBE_DST"

echo
echo "Creating symlinks..."

ln -s "$FFMPEG_SRC" "$FFMPEG_DST"
ln -s "$FFPROBE_SRC" "$FFPROBE_DST"

echo
echo "Symlinks created:"
ls -l "$FFMPEG_DST"
ls -l "$FFPROBE_DST"

echo
echo "Checking zscale support..."

if "$FFMPEG_DST" -filters | grep -q zscale; then
    echo "✔ zscale detected"
else
    echo "⚠ WARNING: zscale NOT detected"
fi

echo
echo "Active ffmpeg version used by cvvdp:"
"$FFMPEG_DST" -version | head -n 3

echo
echo "Setup complete."