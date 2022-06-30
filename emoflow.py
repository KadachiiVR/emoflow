from audioop import avg
from email.mime import audio
from msilib.schema import File
import matplotlib.pyplot as plt                       # Allows you to plot things
import numpy as np      
#print(np.__version__)    
numpy = np

from tqdm import tqdm
#import wave
#import torch
import yaml
import librosa                                        # Python package for music and audio analysis
import librosa.display                                # Allows you to display audio files 
import os                                             # The OS module in Python provides a way of using operating system dependent functionality.
#import scipy.io.wavfile                               # Open a WAV files
                          # Used for working with arrays
import fastai
import glob                                           # Used to return all file paths that match a specific pattern
#import fetch_label                                    # Local class
# Please note: the fetch_label import references a local class that you should define in your local computer. 
# @maheshwari-nikhil on GitHub made this class that you can use: https://github.com/maheshwari-nikhil/emotion-recognition/blob/master/fetch_label.py
#label = fetch_label.FetchLabel()                      # Retrieve files/pathnames matching a specified pattern

# Import fast AI
from fastai import *                                 
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.tabular.all import *
from fastai.text.all import *
#from fastai.vision.widgets import *

# Live Audio
#import struct                                         # Unpack audio data into integers
from PIL import Image as PILImage
import time
from tkinter import DISABLED, TclError
from scipy.fftpack import fft                         # Imports all fft algorithms 
import sounddevice
# import soundfile as sf
from scipy.io.wavfile import write
import argparse
import soundfile as sf
import io
from pythonosc.udp_client import SimpleUDPClient as Osc
import noisereduce as nr
from enum import Enum
from collections import deque

BASEDIR, _ = os.path.split(__file__)


class SimpleTimer():
    def __init__(self, name):
        self.name = name

    def __enter__(self, *args, **kwargs):
        self.starttime = time.time()

    def __exit__(self, *args, **kwargs):
        print(f"{self.name} took {time.time() - self.starttime} seconds")


def read_config(path: string):
    with open(path) as fp:
        cfg = yaml.load(fp, yaml.CLoader)
    cfg['model_path'] = os.path.join(BASEDIR, cfg['model_path'])
    return cfg


def record_wav(seconds: float, fs: float):
    with SimpleTimer("- recording"):
        wav = io.BytesIO()
        recording = sounddevice.rec(int(seconds * fs), samplerate=fs, channels=1)
        sounddevice.wait()
        write(wav, fs, recording)
        wav.seek(0)
        del recording
    return wav


def random_clip2(files, durations, duration):
    total_time = sum(durations)
    tstart = random.random() * (total_time - duration)
    idx = 0
    offset = tstart
    for file, duration in zip(files, durations):
        if (offset < duration):
            break

        idx += 1
        offset -= duration

    print(f"clipping from {files[idx]} at t={offset}")

    with sf.SoundFile(files[idx]) as fp:
        return librosa.load(fp, offset=offset, duration=duration, sr=None)


def random_clip(audio, sr, duration):
    start = random.randint(0, len(audio) - int(sr * duration))
    #print(f"clipping at index {start}")
    return audio[start:start + int(sr * duration)]


def audio_to_image(file, y, sr):
    ax = plt.axes()
    ax.axison = False
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=100)
    melspec = librosa.power_to_db(melspec, ref=np.max)

    librosa.display.specshow(melspec, y_axis=None, x_axis=None, fmax=16000, fmin=64, ax=ax)
    plt.margins(0, tight=true)

    plt.savefig(file, bbox_inches="tight", pad_inches=0, format='jpg')
    #plt.savefig('./tmp/last.png', bbox_inches="tight", pad_inches=0, format='jpg')
    plt.clf()



def train(args):
    config = read_config(args.config)
    print("Config")
    print(config)
    AUDIO_SLICE_DIR = os.path.join(config['tmp'], "clips")
    os.makedirs(AUDIO_SLICE_DIR, exist_ok=True)
    SPECTROGRAM_DIR_TRAIN = os.path.join(config['tmp'], "spectrograms_train")
    os.makedirs(SPECTROGRAM_DIR_TRAIN, exist_ok=True)
    SPECTROGRAM_DIR_TEST = os.path.join(config['tmp'], "spectrograms_test")
    os.makedirs(SPECTROGRAM_DIR_TEST, exist_ok=True)
    random.seed(config['random_seed'])

    if not(args.skip_img_gen):
        for labelid, label in config['labels'].items():
            print(f"generating data for label {label}")
            audio_bucket = os.path.join(AUDIO_SLICE_DIR, label)
            os.makedirs(audio_bucket, exist_ok=True)
            train_bucket = os.path.join(SPECTROGRAM_DIR_TRAIN, label)
            os.makedirs(train_bucket, exist_ok=True)
            test_bucket = os.path.join(SPECTROGRAM_DIR_TEST, label)
            os.makedirs(test_bucket, exist_ok=True)
            data_dir = os.path.join(config['dataset'], label)

            files = glob.glob(data_dir+"/*")
            #durations = [librosa.get_duration(filename=file) for file in files]

            all_audio = None
            sr = None
            for file in files:
                with sf.SoundFile(file) as fp:
                    y, xsr = librosa.load(fp, sr=None)
                    if sr is None:
                        sr = xsr
                    else:
                        assert xsr == sr
                if all_audio is None:
                    all_audio = y
                else:
                    all_audio = np.append(all_audio, y)

            for idx in tqdm(range(config['sample_count'])):
                y = random_clip(all_audio, sr, config['sample_len'])

                if config['noise_reduction'] > .001:
                    yt = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=config['noise_reduction'])
                else:
                    yt = y

                yt, _ = librosa.effects.trim(yt)

                if idx % 8 == 0:
                    imgpath = os.path.join(test_bucket, "{}{}.jpg".format(label, str(idx).zfill(6)))
                else:
                    imgpath = os.path.join(train_bucket, "{}{}.jpg".format(label, str(idx).zfill(6)))

                audio_to_image(imgpath, yt, sr)

            print("done", label)
        print("generated spectrograms!")

    dls = ImageDataLoaders.from_folder(SPECTROGRAM_DIR_TRAIN, valid_pct=0.2, seed=42, num_workers=6, bs=16)
    print(dls.vocab.o2i)
    learn = vision_learner(
        dls,
        models.resnet34,
        normalize=False,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        path=config['model_path'],
        model_dir=config['model_dir']
    )

    print("LEARNING!")
    x = learn.lr_find()
    lr = x.valley
    lfr = learn.fit(10, float(f"{lr:.2e}"))
    learn.show_results()

    interp = ClassificationInterpretation.from_learner(learn)
    losses, idxs = interp.top_losses()

    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

    plt.show()
    learn.freeze()
    learn.export(config['model_export_path'])


#DEFAULT_EMOTES = {"neutral": 1, "excited": 0, "laugh": 0}



def wav2imgdata(wav: File, nrfactor: float):
    with SimpleTimer("- processing"):
        y, sr = librosa.load(wav)
        if nrfactor > .001:
            yt = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=nrfactor)
        else:
            yt = y

        yt,_=librosa.effects.trim(yt)

        # Converting the sound clips into a melspectogram with librosa
        # A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale
        #audio_spectogram = librosa.feature.melspectrogram(y=yt, sr=sr, n_fft=1024, hop_length=100)
        power = librosa.feature.rms(y=yt)
        power = np.average(power)
        #print(f"Sample RMS was {power}")

        # Convert a power spectrogram (amplitude squared) to decibel (dB) units with power_to_db
        #audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)

        # Display the spectrogram with specshow
        #librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')

        #p = os.path.join("C:\\Users\\tobia\\Documents\\projects\\feelings_detector\\live_images", "{}.jpg".format(str(count)))

        with io.BytesIO() as imgfile:
            audio_to_image(imgfile, yt, sr)
            imgfile.seek(0)
            #imgdata = plt.imread(imgfile, format='jpg')
            img = fastai.vision.core.PILImage.create(imgfile)

        plt.clf()
        del y
        del yt
        #del audio_spectogram
    return img, power


# def old_interpret_scores(window, weights, weight, db, emotes):
#     sum_weights = sum(list(weights))
#     print(sum_weights)

#     smooth_emotes = {
#         k: sum([
#             d[k] * w
#             for d, w
#             in zip(list(window), list(weights))
#         ]) / (sum_weights + 5)
#         for k
#         in emotes.keys()
#     }
#     print(emotes, smooth_emotes, weight, db)

#     highest = max(smooth_emotes, key=smooth_emotes.get)

#     NEUTRAL = {"HappyHigh": .01, "HappyLow": .01, "NegativeHigh": .01, "NegativeLow": .01}
#     EXCITED = {"HappyHigh": .9, "HappyLow": .01, "NegativeHigh": .01, "NegativeLow": .01}
#     SAD = {"HappyHigh": .01, "HappyLow": .01, "NegativeHigh": .01, "NegativeLow": .9}

#     if sum_weights < 5 or weight < 0.1:
#         return NEUTRAL
#     elif highest == "excited":
#         return EXCITED
#     elif smooth_emotes["sad"] > 0.85:
#         return SAD
#     elif smooth_emotes["excited"] > .2:
#         return EXCITED
#     else:
#         return NEUTRAL


def interpret_scores(window, weights, weight, db, emotes):
    sum_weights = sum(list(weights))
    print(sum_weights)

    smooth_emotes = {
        k: sum([
            d[k] * w
            for d, w
            in zip(list(window), list(weights))
        ]) / (sum_weights + 5)
        for k
        in emotes.keys()
    }
    for k, v in sorted(emotes.items()):
        print("%s: %.3f | %.3f" % (k, v, (smooth_emotes[k])))
    print(weight, sum_weights, db)
    
    highest = max(smooth_emotes, key=smooth_emotes.get)

    NEUTRAL = 0
    EXCITED = 1
    LAUGH = 2
    REVERSE = {
        NEUTRAL: "NEUTRAL",
        EXCITED: "EXCITED",
        LAUGH: "LAUGH"
    }

    emotion = NEUTRAL
    weight = 0.9

    if sum_weights < 5 or weight < 0.1:
        emotion = NEUTRAL
        weight = 0.9
    elif highest == "excited":
        emotion = EXCITED
        weight = 0.9
    elif smooth_emotes["laugh"] > 0.75 or (smooth_emotes["laugh"] > .3 and emotes["laugh"] > .9):
        emotion = LAUGH
        weight = 0.9
    elif smooth_emotes["excited"] > .3:
        emotion = EXCITED
        weight = 0.9
    else:
        emotion = NEUTRAL
        weight = 0.9

    print(f"Sending Emotion: {emotion} ({REVERSE[emotion]}), Weight: {weight}")

    return emotion, weight
    


def run(args):
    config = read_config(args.config)
    catsdir = os.path.join(config['tmp'], 'spectrograms_train')
    dls = ImageDataLoaders.from_folder(catsdir, valid_pct=0.2, seed=42, num_workers=0, bs=16)
    print(dls.vocab.o2i)
    print(dls.vocab)
    model = vision_learner(
        dls,
        models.resnet34,
        normalize=False,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        path=config['model_path'],
        model_dir=config['model_dir']
    )
    print("Model loaded")

    DEFAULT_EMOTES = {
        v: int(k == 0)
        for k, v
        in config['labels'].items()
    }

    osc = Osc('127.0.0.1', args.port)
    print("OSC connected")

    window = deque(maxlen=config['window'])
    weights = deque(maxlen=config['window'])
    for _ in range(config['window']):
        window.append({
            k: 0
            for k
            in dls.vocab
        })
        weights.append(0)

    while(True):
        with record_wav(config['sample_len'], config['fs']) as wav:
            in_data, rms = wav2imgdata(wav, config['noise_reduction'])
            with SimpleTimer('- predicting'):
                label, tn, probs = model.predict(in_data)
            probs = list(probs)

            emotes = {
                k: float(probs[dls.vocab.o2i[k]])
                for k
                in dls.vocab
            }

            db = 10 * np.log10(rms)
            weight = max(db - (-23), 0) 

            if weight < 1:
                emotes = DEFAULT_EMOTES

            window.append(emotes)
            weights.append(weight)

            emotion, emotion_weight = interpret_scores(window, weights, weight, db, emotes)
            osc.send_message(f"{config['paramprefix']}Emotion", emotion)
            osc.send_message(f"{config['paramprefix']}EmotionWeight", emotion_weight + .001)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--config", type=str)
    parser_train.add_argument("--skip-img-gen", action="store_true")
    parser_run = subparsers.add_parser("run")
    parser_run.add_argument("--config", type=str)
    parser_run.add_argument("--port", type=int, default=9000)

    args = parser.parse_args()

    if args.subparser_name == "train":
        train(args)
    elif args.subparser_name == "run":
        run(args)


if __name__ == "__main__":
    main()