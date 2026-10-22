# Voice prompts

`taiwan_speaker.wav`: the example speaker of MediaTek Research's BreezyVoice
(https://github.com/mtkresearch/BreezyVoice, `data/example.wav`, Apache License 2.0), copied
unchanged. The cloned "台灣腔" voices (Qwen3-TTS, VoxCPM2, Breeze-TTS-2 through mlx-audio,
`src/local_llm/mlx_tts.py`) use it with its transcript (`neural_tts.PROMPT_TEXT`) so that they
speak with a Taiwanese accent. It is used to synthesise test speech only.
