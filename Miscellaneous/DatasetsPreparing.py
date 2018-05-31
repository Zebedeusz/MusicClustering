from __future__ import print_function

import contextlib
import os
import sys
import wave

import audioread


# decodes file on given path to .wav with 1 channel
def decode_au(filename):
    filename = os.path.abspath(os.path.expanduser(filename))
    format = filename.split(".")[-1]
    if not os.path.exists(filename):
        print("File not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with audioread.audio_open(filename) as f:
            print('Input file: %i channels at %i Hz; %.1f seconds.' %
                  (f.channels, f.samplerate, f.duration),
                  file=sys.stderr)
            print('Backend:', str(type(f).__module__).split('.')[1],
                  file=sys.stderr)

            with contextlib.closing(wave.open(filename.replace("." + format, ".wav"), 'w')) as of:
                of.setnchannels(1)
                of.setframerate(f.samplerate)
                of.setsampwidth(2)

                for buf in f:
                    of.writeframes(buf)

    except audioread.DecodeError:
        print("File could not be decoded.", file=sys.stderr)
        sys.exit(1)


# decodes file on given path to .wav with 1 channel
def decode_mp3(filename):
    filename = os.path.abspath(os.path.expanduser(filename))
    format = filename.split(".")[-1]
    if not os.path.exists(filename):
        print("File not found.", file=sys.stderr)
        sys.exit(1)

    from pydub import AudioSegment
    sound = AudioSegment.from_mp3(filename)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(22050)
    sound.export(filename.replace("." + format, ".wav"), format="wav")


# decodes file on given path to .wav with 1 channel
def decode_wav(filename):
    filename = os.path.abspath(os.path.expanduser(filename))
    format = filename.split(".")[-1]
    if not os.path.exists(filename):
        print("File not found.", file=sys.stderr)
        sys.exit(1)

    from pydub import AudioSegment
    sound = AudioSegment.from_wav(filename)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(22050)
    sound.export(filename.replace("." + format, "_ch1_fr22500.wav"), format="wav")


def prepare_files_in_dir(directory):
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for filename in filenames:
            print("\nDecoding file " + filename)
            format = filename.split(".")[-1]
            if format == "au":
                decode_au("{}/{}".format(directory, filename))
            elif format == "mp3":
                decode_mp3("{}/{}".format(directory, filename))
            elif format == "wav":
                decode_wav("{}/{}".format(directory, filename))
            else:
                print("Format not recognised")
                print(format)


if __name__ == '__main__':
    directory = "/media/michal/HDD1/Music Emotion Datasets/1000songs/clips_45seconds/"
    # for (dirpath, dirnames, filenames) in os.walk(directory):
    #     for dirname in dirnames:
    #         path = directory + "/" + dirname
    prepare_files_in_dir(directory)

    # decode("/media/michal/HDD1/Music Emotion Datasets/genres/genres/blues/blues.00000.au")
