#!/bin/bash

# Download the datasets
mkdir -p /home/$(id -un)/src/data
cd /home/$(id -un)/src/data
echo "Downloading AIDER"
wget https://zenodo.org/records/3888300/files/AIDER.zip
mkdir -p AIDERV2
cd AIDERV2
echo "Downlaoding AIDERV2 Test"
wget https://zenodo.org/records/10891054/files/Test.zip
echo "Downloading AIDERV2 Train"
wget https://zenodo.org/records/10891054/files/Train.zip
echo "Downloading AIDERV2 Val"
wget https://zenodo.org/records/10891054/files/Val.zip
echo "Unzip AIDERV2 Test"
unzip Test.zip
echo "Unzip AIDERV2 Train"
unzip Train.zip
echo "Unzip AIDERV2 Val"
unzip Val.zip
cd ..
echo "Unzip AIDER"
unzip AIDER.zip
cd ..
