Solution overview
---

I used Encoder-Decoder Wavenet architecture similar to [@sjv](https://www.kaggle.com/seanjv)'s awesome solution for [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting).  
I implemented it by [Chainer](https://docs.chainer.org/en/stable/index.html)

Setup
---

1. Run `setup_dirs.sh` to make directories.
2. Download dataset and put them into `data/input/`.
3. Download weather data from [huntermcgushion/rrv-weather-data](https://www.kaggle.com/huntermcgushion/rrv-weather-data).  
   Extract `1-1-16_5-31-17_Weather.zip` to a directory `1-1-16_5-31-17_Weather`.  
   Put the directory and `weather_stations.csv` into `data/input/`.

How to run
---

Preprocess the weather data.

```
python prepare_weather.py
```

Run the training script. 

```
python seq_run.py
```

It calls `run.py` several times with some configurations and various seeds.  
It saves results as one folder per one run in `data/output/`.  
After the end of training, gather the result folders into one folder.  
Then run the ensemble script.

```
python ensemble.py --target_dir "path/to/dir/" --without_valid
```

References
---
- sjv's original implementation  
  [https://github.com/sjvasquez/web-traffic-forecasting](https://github.com/sjvasquez/web-traffic-forecasting)  
- WaveNet: A Generative Model for Raw Audio
  [https://deepmind.com/blog/wavenet-generative-model-raw-audio/](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)  