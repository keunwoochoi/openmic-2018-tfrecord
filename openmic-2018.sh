#!/usr/bin/env bash

cd ~
git clone https://github.com/keunwoochoi/openmic-2018-tfrecord.git
# download openmic-2018

mkdir 'openmic-2018-raw'
wget https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz
tar xvzf openmic-2018-v1.0.0.tgz -C openmic-2018-raw

# folder to save tfrecord files
mkdir 'openmic-2018'

cd ~/openmic-2018-tfrecord
pip install -r requirements.txt

python openmic2018.py --openmic_dir=$HOME"/openmic-2018-raw/openmic-2018" \
  --tfrecord_dir=$HOME"/openmic-2018" \
  --sampling_rate=22050
