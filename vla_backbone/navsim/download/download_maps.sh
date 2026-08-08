#!/bin/bash

DEST="./dataset/nuplan"

mkdir -p "$DEST"
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip -P "$DEST"
unzip "$DEST/nuplan-maps-v1.1.zip" -d "$DEST"
rm "$DEST/nuplan-maps-v1.1.zip"

if [ -d "$DEST/nuplan-maps-v1.0" ]; then
    mv "$DEST/nuplan-maps-v1.0" "$DEST/maps"
    echo "Success: Maps moved to $DEST/maps"
else
    echo "Error: Directory $DEST/nuplan-maps-v1.0 not found."
fi