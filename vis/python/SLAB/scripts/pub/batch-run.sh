#!/bin/bash

DIR="$1"
N="$2"

if [ $# -ne 2 ]; then
  echo "Usage: $0 <dir> <num_processes>"
  exit 1
fi

if [ ! -d "$DIR" ]; then
  echo "Error: '$DIR' is not a directory"
  exit 1
fi

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -le 0 ]; then
  echo "Error: num_processes must be a positive integer"
  exit 1
fi

DIR_ABS="$(cd "$DIR" && pwd)"

find "$DIR_ABS" -type f \( -name "*.athdf" \) -print0 \
| xargs -0 -P "$N" -n 1 -I {} python plot_slice.py --file "{}"