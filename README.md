# PFR-MWM
Annual project at Telecom Paris with MWM. The goal is to develop a framework that creates a music video automatically given video shots and a music.

## Model download

### PGL-SUM

The following is inspired from the PGL-SUM github : https://github.com/e-apostolidis/PGL-SUM

We have released the [**trained models**](https://doi.org/10.5281/zenodo.5635735) for our main experiments -namely `Table III` and `Table IV`- of our ISM 2021 paper. The [`inference.py`](inference/inference.py) script, lets you evaluate the -reported- trained models, for our 5 randomly-created data splits. Firstly, download the trained models, with the following script:
``` bash
cd Video/models/scoring_models/PGL-SUM
sudo apt-get install unzip wget
wget "https://zenodo.org/record/5635735/files/pretrained_models.zip?download=1" -O pretrained_models.zip
unzip pretrained_models.zip -d inference
rm -f pretrained_models.zip
```
