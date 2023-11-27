import os
import gc
import sys
import argparse
import tempfile
from glob import glob

from pydub import AudioSegment
from moviepy.editor import *
import demucs.separate
import whisperx
import torch
import librosa
import numpy as np
from translatepy import Translator
from scipy.io.wavfile import write as write_wav
from scipy.spatial.distance import hamming

sys.path.append(os.path.join(sys.path[0], "VALL-E-X"))
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from utils.prompt_making import make_prompt

parser = argparse.ArgumentParser()
parser.add_argument("video", help="path of the target video file", type=str)
parser.add_argument("hf_auth_token", help=("access token generated from `huggingface.co` "
                                           "(readonly will work, write is not required)"),
                    type=str)
parser.add_argument("--srt", help="path of the translated subtitle file", type=str)
parser.add_argument("-v", "--verbose", help="path of the translated subtitle file",
                    default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--whisperx_batch_size", help=("batch size for WhisperX inference, "
                                                   "reduce if low on GPU mem"),
                    default=16, type=int)
parser.add_argument("--whisperx_compute_type", help=("precicient for WhisperX inference, "
                                                     "default float16, "
                                                     "reduce to float32 if on CPU. "
                                                     "you may change to \"int8\" "
                                                     "if low on GPU mem (may reduce accuracy)")
                    default="float16", type=str)
# support only English, Chinese, and Japanese
parser.add_argument("--target_language", help="target language to translate to",
                   choices=['English', 'Chinese', 'Japanese'], default='中国語', type=str)
parser.add_argument("--autocheck", help="use a model to do quality control generate better audio",
                    default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--postpro_trim_silence", help="trim silence in generated audio",
                    default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--postpro_time_stretch", help="fit generated audio to the time slot of the subtitle",
                    default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("-o", "--output", help="path for the output file", default=None, type=str)
args = parser.parse_args()

assert args.target_language in ['English', 'Chinese', 'Japanese', '中国語']
target_lang_code = {'English': 'en',
                    'Chinese': 'zh',
                    'Japanese': 'ja',
                    '中国語': 'zh'}
AUTOCHECK_MAX_RETRY = 3

## audio utils for post-production merge audio to video #####
def match_target_sound_amplitude(sound, target_sound):
    change_in_dBFS = target_sound.dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

## audio utils to enhance audio for prompting ###############
def match_target_amplitude(sound, target_dBFS=-16):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

## audio utils to split audio to chunks ####################
def trim_audio(intervals, input_file_path, output_file_path):
    # load the audio file
    audio = AudioSegment.from_file(input_file_path)

    # iterate over the list of time intervals
    for i, (start_time, end_time) in enumerate(intervals):
        # extract the segment of the audio
        segment = audio[start_time*1000:end_time*1000]
        segment = match_target_amplitude(segment, -14)

        # construct the output file path
        output_file_path_i = f"{output_file_path}_{i}.wav"

        # export the segment to a file
        segment.export(output_file_path_i, format='wav')

## find leading silence of audio to trim ###################
def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

## find tailing silence of audio to trim ###################
def detect_tailing_silence(sound, silence_threshold=-50.0, chunk_size=-10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = -1 # ms

    assert chunk_size < 0 # to avoid infinite loop
    while sound[trim_ms+chunk_size:trim_ms].dBFS < silence_threshold and abs(trim_ms) < len(sound):
        trim_ms += chunk_size

    return trim_ms


def main(args):
    temp_dir = tempfile.TemporaryDirectory()
    if args.verbose:
        print("Creating tmp dir", temp_dir.name)

    audio_file = os.path.join(temp_dir.name, "test.wav")

    videoclip = VideoFileClip(args.video)
    audioclip = videoclip.audio
    audioclip.write_audiofile(audio_file)
    
    demucs.separate.main(["--two-stems", "vocals",
                          "-o", os.path.join(temp_dir.name, separated),
                          "-n", "mdx_extra",
                          audio_file])
    
    for filename in glob(f"{temp_dir.name}/**/*.wav", recursive=True):
        os.rename(filename,
                  os.path.join(temp_dir.name,
                               os.path.basename(filename)))

    if args.verbose:
        print("Complex analyzing on audio data....")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.whisperx_batch_size
    compute_type = args.whisperx_compute_type if torch.cuda.is_available() else "float32"

    audio = whisperx.load_audio(audio_file)

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=args.hf_auth_token,
                                                 device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    # result = whisperx.assign_word_speakers(diarize_segments, result)
    if args.verbose:
        print(diarize_segments)
    # print(result["segments"]) # segments are now assigned speaker IDs

    # delete model if low on GPU resources
    del diarize_model
    gc.collect()
    torch.cuda.empty_cache()

    window_list = []
    for sentence in diarize_segments["segment"]:
        window_list.append([sentence.start, sentence.end])

    if args.verbose:
        print("Timestamp of each subtitle lines recorded.")

    # test it out
    if args.verbose:
        print("Trimming audio...")
    trim_audio(window_list,
               os.path.join(temp_dir.name, "vocals.wav"),
               os.path.join(temp_dir.name, "test_output"))
    if args.verbose:
        print("...done! <3")

    model = whisperx.load_model("medium",
                                device,
                                compute_type=compute_type)

    # detect the spoken language
    audio = whisperx.load_audio(audio_file)
    language = model.detect_language(audio)

    text_list = []
    for i in range(len(window_list)):
        audio_file = f"{temp_dir.name}/test_output_{i}.wav"
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio,
                                  batch_size=batch_size,
                                  language=language)
        texts = [seg["text"] for seg in result["segments"]]
        text_list.append(','.join(texts))

    # delete model if low on GPU resources
    if not args.autocheck:
        torch.cuda.empty_cache()
        del model
        gc.collect()

    translated_text = []
    if args.srt == None or len(args.srt) == 0:
        translator = Translator()

        for text in text_list:
            outputs = translator.translate(text, args.target_language)
            translated_text.append(outputs.result)

    else:
        raise NotImplementedError("Sorry: not implemented yet!")

    # voice cloning
    for i, text in enumerate(text_list):
        make_prompt(name=f"speecher{i}",
                    audio_prompt_path=f"{temp_dir.name}/test_output_{i}.wav",
                    transcript=text)

    # download and load all models
    preload_models()

    for i, text_prompt in enumerate(translated_text):
        # generate audio from text
        audio_array = generate_audio(text_prompt, prompt=f"speecher{i}")
    
        # save audio to disk
        write_wav(os.path.join(temp_dir.name, f"vallex_generation_{i}.wav"),
                  SAMPLE_RATE, audio_array)
        if args.autocheck:
            check_record = []
            language = target_lang_code[args.target_language]
            os.rename(os.path.join(temp_dir.name, f"vallex_generation_{i}.wav"),
                      os.path.join(temp_dir.name, f"vallex_generation_{i}_0.wav"))
            for retry in range(AUTOCHECK_MAX_RETRY):
                audio_file = os.path.join(temp_dir.name,
                                          f"vallex_generation_{i}_{retry}.wav")
                audio = whisperx.load_audio(audio_file)
                tg = model.transcribe(audio,
                                      batch_size=batch_size,
                                      language=language)
                dist = 0
                while len(tg) != len(text_prompt):
                    if len(text_prompt) > len(tg):
                        tg += ' '
                    else:
                        dist += 1
                        tg.pop(-1)
    
                dist += int(hamming(tg, text_prompt) * len(tg))
                check_record.append(dist)
                if dist == 0:
                    os.rename(os.path.join(temp_dir.name,
                                           f"vallex_generation_{i}_{retry}.wav"),
                              os.path.join(temp_dir.name, f"vallex_generation_{i}.wav"))
                    break
                elif retry + 1 < AUTOCHECK_MAX_RETRY:
                    # generate audio from text
                    audio_array = generate_audio(text_prompt, prompt=f"speecher{i}")
                
                    # save audio to disk
                    write_wav(os.path.join(temp_dir.name,
                                           f"vallex_generation_{i}_{retry + 1}.wav"),
                              SAMPLE_RATE, audio_array)
                else:
                    j = check_record.index(min(check_record))
                    os.rename(os.path.join(temp_dir.name, f"vallex_generation_{i}_{j}.wav"),
                              os.path.join(temp_dir.name, f"vallex_generation_{i}.wav"))

            torch.cuda.empty_cache()
            del model
            gc.collect()

    sound0 = AudioSegment.from_file(os.path.join(temp_dir.name, "vocals.wav"))
    sound1 = AudioSegment.from_file(os.path.join(temp_dir.name, "no_vocals.wav"))
    for i, timestamps in enumerate(window_list):
        start_time, end_time = timestamps
        original_sound = sound0[start_time*1000:end_time*1000]
        sound2 = AudioSegment.from_file(os.path.join(temp_dir.name,
                                                     f"vallex_generation_{i}.wav"))

        # pydub trim audio silence head/tail
        if args.postpro_trim_silence:
            start_time = max(start_time - detect_leading_silence(sound2) / 1000, 0)
            sound2 = sound2[detect_leading_silence(sound2):]
            sound2 = sound2[:detect_tailing_silence(sound2)]

        # librosa align audio duration
        if args.postpro_time_stretch:
            duration = detect_tailing_silence(sound2) - detect_leading_silence(sound2)
            if duration > len(original_sound):
                speed  = duration / len(original_sound)
                sample = np.asarray(sound2.get_array_of_samples(),
                                    dtype=np.float32)
                sample /= sample.max() * 1.5
                sample = librosa.effects.time_stretch(sample, rate=speed)
                sample = np.array(sample * (1<<15), dtype=np.int16)
                sound2 = AudioSegment(sample.tobytes(),
                                      frame_rate=sound2.frame_rate,
                                      sample_width=sample.dtype.itemsize,
                                      channels=1)

        # pydub align audio amplitude
        sound2 = match_target_sound_amplitude(sound2, original_sound)

        # pydub put audio to timestamp
        sound1 = sound1.overlay(sound2, start_time*1000)

    sound1.export(os.path.join(temp_dir.name, "new-audio.wav"), format='wav')

    new_audioclip = AudioFileClip(os.path.join(temp_dir.name, "new-audio.wav"))
    videoclip = videoclip.set_audio(new_audioclip)
    videoclip.write_videofile(os.path.join(temp_dir.name, "new-video.mp4"))

    output_path = args.output
    if args.output is None:
        output_path = os.path.join(os.path.split(args.video)[0], "new-video.mp4")
    os.rename(os.path.join(temp_dir.name, "new-video.mp4"), output_path)

    temp_dir.cleanup()
    if args.verbose:
        print("Creating tmp dir", temp_dir.name)

if __name__ == '__main__':
    main()
