# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0
#!/bin/bash

set -euo pipefail

URL="https://alldxpprdstor.blob.core.windows.net/large-content-media/LianderPower.zip"
DEST="data/LianderPower"
ZIP="data/LianderPower.zip"

mkdir -p "$DEST"

if [ -f "$ZIP" ] && unzip -t "$ZIP" &>/dev/null; then
    echo "LianderPower.zip already exists and is valid, skipping download."
else
    echo "Downloading LianderPower.zip..."
    curl -L -o "$ZIP" "$URL"
fi

echo "Extracting data files..."
unzip -q "$ZIP" \
    "LianderPower/data/croissant.json" \
    "LianderPower/data/measurements.parquet" \
    "LianderPower/data/weather.parquet" \
    -d /tmp/LianderPower_extract

echo "Moving files to $DEST..."
mv /tmp/LianderPower_extract/LianderPower/data/croissant.json "$DEST/"
mv /tmp/LianderPower_extract/LianderPower/data/measurements.parquet "$DEST/"
mv /tmp/LianderPower_extract/LianderPower/data/weather.parquet "$DEST/"

echo "Cleaning up temp files..."
rm -rf /tmp/LianderPower_extract

echo "Done! Data available at $DEST"
