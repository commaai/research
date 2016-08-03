#!/bin/bash
mkdir -p dataset/camera dataset/log
if [ -a comma-dataset.zip ]
then
    wget -c https://archive.org/download/comma-dataset/comma-dataset.zip
else
    wget -nc https://archive.org/download/comma-dataset/comma-dataset.zip
fi

mkdir -p dataset
cd dataset
unzip ../comma-dataset.zip
