#!/bin/bash
mkdir -p dataset/camera dataset/log

wget --no-check-certificate --continue https://archive.org/download/comma-dataset/comma-dataset.zip
mkdir -p dataset
cd dataset
unzip ../comma-dataset.zip
