# anime-translator
Applying deep learning to translate animation and re-generate audio.

Supported output languages:
| Chinese (Mandarin) | Japanese | English |
|-|-|-|

|[![YouTube Demo](https://img.youtube.com/vi/Rl5Z85zWLgk/0.jpg)](https://youtu.be/Rl5Z85zWLgk)|
|-|
|Click to view [Demo video (Japanese to Chinese) on YouTube](https://youtu.be/Rl5Z85zWLgk)|

## Usage
Input video and translate it to Chinese (Mandarin):
```bash
python3 convert.py ~/Desktop/test.mp4 --target_language Chinese
```

Input video, subtitle file (`*.srt`) and translate it to Chinese (Mandarin):
```bash
python3 convert.py ~/Desktop/test.mp4 --srt ~/Desktop/test.srt --target_language Chinese
```
> ⚠️ Parsing SRT file is on Todo list. This feature is not available.

## Get started
```bash
git clone https://github.com/NewJerseyStyle/anime-translator.git
cd anime-translator
pip install -r requirements.txt
python3 convert.py ~/Desktop/test.mp4
```

## Credits
Credits to powerful tools made this repo possible:
- [MoviePy](https://github.com/Zulko/moviepy)
- [whisperX project](https://github.com/m-bain/whisperX)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Microsoft VALL-E (X)](https://github.com/Plachtaa/VALL-E-X)
- [PyDub](https://github.com/jiaaro/pydub)
- [librosa](https://github.com/librosa/librosa)
- [Facebook Demucs](https://github.com/facebookresearch/demucs)
- [translatepy](https://github.com/Animenosekai/translate)

## Known issues
- `*.srt` file timeframe and generated audio timeframe alignment problem
