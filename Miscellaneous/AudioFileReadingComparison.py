import scipy.io.wavfile as wav
import numpy
from pydub import AudioSegment
from matplotlib import pyplot as plt


filepath = "/media/michal/HDD/Music Emotion Datasets/Eerola/dataverse_files/stimuli/test-stimuli-200-2009-05-29/006.wav"
filepath = "/media/michal/HDD/Datasets/1000_songs_dataset/clips_45seconds/116.mp3.wav"

#1
print("Opening with scipy.io.wavfile")

f, sound = wav.read(filepath)

print("type: {}".format(type(sound)))
print("len: {}".format(len(sound)))
print("numpy.shape: {}".format(numpy.shape(sound)))
print("frames 121:160: {}".format(sound[121:160]))
print("max frame value: {}".format(max(sound)))
print("min frame value: {}\n".format(min(sound)))
plt.hist(sound, numpy.arange(int(min(sound)), int(max(sound)), 20))
plt.show()


#2
print("Opening with AudioSegment")

if str(filepath).endswith(".wav"):
    sound = AudioSegment.from_wav(filepath).raw_data
elif str(filepath).endswith(".mp3"):
    sound = AudioSegment.from_mp3(filepath).raw_data

print("type: {}".format(type(sound)))
print("len: {}".format(len(sound)))
print("frame 121: {}".format(sound[121]))
print("max frame value: {}".format(max(sound)))
print("min frame value: {}\n".format(min(sound)))

#3

print("Opening with wave")
import wave
file = wave.open(filepath)
sound = file.readframes(file.getnframes())

print("type: {}".format(type(sound)))
print("len: {}".format(len(sound)))
print("channels: {}".format(file.getnchannels()))
print("frames: {}".format(file.getnframes()))
print("framerate: {}".format(file.getframerate()))
print("frames 1000000: {}".format(sound[1000000]))
print("max frame value: {}".format(max(sound)))
print("min frame value: {}\n".format(min(sound)))
sound = numpy.asarray(sound,dtype=numpy.dtype('b'))
plt.hist(sound, numpy.arange(int(min(sound)), int(max(sound)), 20))
plt.show()
