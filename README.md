# openmic-2018-tfrecord

[OpenMIC-2018](https://zenodo.org/record/1432913#.Xpi4Ny-ZPVv) is a dataset for instrument ID. 
This repo contains scripts to create its `tfrecords` version.


## Usage

### Preparation

As written in `openmic-2018.sh`.

```bash
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

```

### Loading

See `dataset.py`.

```python
openmic = dataset.Openmic(path='openmic-2018', train=True)
ds = openmic.get_dataset(batch_size=16, shuffle=True, repeats=-1)

# now use `ds` to train your model  
```


## Notes
Instruments are alphabetically sorted.
```
{"accordion": 0, "banjo": 1, "bass": 2, "cello": 3, "clarinet": 4, 
"cymbals": 5, "drums": 6, "flute": 7, "guitar": 8, "mallet_percussion": 9, 
"mandolin": 10, "organ": 11, "piano": 12, "saxophone": 13, "synthesizer": 14, 
"trombone": 15, "trumpet": 16, "ukulele": 17, "violin": 18, "voice": 19}
```

Inside the resulting `tfrecord` files:
```python
>>> it = tf_record_iterator('train_0.tfrecords')
>>> d = next(it)
>>> d
{'track_id': [b'000046_3840'], 
'inst_idxs': array([ 4,  7, 16]), 
'waveform_22050': array([ 0.        ,  0.        ,  0.        , ..., -0.01253502, -0.01636652, -0.02074894], dtype=float32), 
'nhot_vector': array([0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], dtype=float32)}
```
