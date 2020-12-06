# german-tts
German Tacotron 2 and Multi-band MelGAN in TensorFlow with TF Lite inference support

## Overview
I am releasing pretrained German neural text-to-speech (TTS) models Tacotron 2 and Multi-band MelGAN. It supports inference with `saved_model` and `TF Lite` formats.

- See [`inference.py`](https://github.com/monatis/german-tts/blob/main/inference.py) to infer with `saved_model`.
- See [`inference_tflite.py`](https://github.com/monatis/german-tts/blob/main/inference_tflite.py) to infer with `TF Lite`.
- See [`e2e-notebook.ipynb`](https://github.com/monatis/german-tts/blob/main/e2e-notebook.ipynb) to check how I exported to these model formats.
- see [releases](https://github.com/monatis/german-tts/releases) to download pretrained models.

## Dataset
I trained these models on [Thorsten dataset](https://github.com/thorstenMueller/deep-learning-german-tts) by Thorsten Müller. It is licensed under the terms of Creative Commons Zero V1 Universal (CC0), which is used to opt out of copyright entirely and ensure that the work has the widest reach. Thanks [@thorstenMueller](https://github.com/thorstenMueller) for such a great contribution to the community.

## Notes
Some good guys are doing a great job at [tensorspeech/TensorFlowTTS](https://github.com/tensorspeech/TensorFlowTTS), which was already supporting TTS in English, Chinese and Korean. I wanted to contribute with support for German and trained these models. Now it supports both training and inference with proper processors. A detailed blog post will follow up, but some quick notes for now:

- I made use of [german_transliterate](https://github.com/repodiac/german_transliterate) For text preprocessing. Basically it normalizes numbers (e.g. converts digits to words), expands abbreviations and cares German umlauts and punctuations. For inference examples released in this repo, it is the only dependency apart from TensorFlow.
- You need to convert input text to numerical IDs to feed into the model. I am sharing a reference implementation for this in inference examples, and you need to code this logic to use the models in non-Python environments (e.g., Android).
- `Tacotron 2` produces some noise at the end, and you need to cut it off. Again, inference examples show how to do this.
- I exported `Multi-band MelGAN` to `TF Lite` without optimizations because it produced some background noise when I exported with the default ones. I used default optimizations in `Tacotron 2`.
- `saved_model` formats that I am releasing here are not suitable for finetuning. Architecture implementation uses `Subclassing API` in TensorFlow 2.x and gets multiple inputs in `call` method for teacher forcing during training. This caused some problems when exporting to `saved_model` and I had to remove this logic before exporting. If you want to finetune models, please see [my fork of TensorFlowTTS](https://github.com/monatis/TensorFlowTTS).

## License
You can use these pretrained model artifacts and code examples under the terms of Apache 2.0 license. On the other hand, you may want to contact me for paid consultancies and/or collaborations in speech and/or NLP projects at the email address shown on [my profile](https://github.com/monatis).