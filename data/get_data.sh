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


echo "Downloading Liander2024 benchmark..."
# Liander2024 data 
uv run hf download OpenSTEF/liander2024-energy-forecasting-benchmark --repo-type dataset --local-dir data/liander2024

echo "Downloading the fev-bench energy datasets..."
# fev-bench energy tasks (the 26 Energy-domain tasks). The benchmark streams these from the
# Hugging Face Hub by default; this local mirror serves machines without Hub access when
# `datasets_prefix = "data/fev_datasets"` is set in the FevBenchBenchmark config section.
FEV_CONFIGS="proenfo_gfc12 proenfo_gfc14 proenfo_gfc17 entsoe_15T entsoe_30T entsoe_1H epf_be epf_de epf_fr epf_np epf_pjm \
  solar_with_weather_15T solar_with_weather_1H solar_1D solar_1W kdd_cup_2022_10T kdd_cup_2022_30T kdd_cup_2022_1D \
  ETT_15T ETT_1D ETT_1H ETT_1W ercot_1H ercot_1D ercot_1W ercot_1M"
FEV_INCLUDES=""
for cfg in $FEV_CONFIGS; do FEV_INCLUDES="$FEV_INCLUDES --include $cfg/*"; done
uv run hf download autogluon/fev_datasets --repo-type dataset --local-dir data/fev_datasets $FEV_INCLUDES

echo "Done! Data available at $DEST"
