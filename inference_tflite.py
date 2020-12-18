# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perform preprocessing and raw feature extraction for LJSpeech dataset."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import re
import time
from scipy.io import wavfile
from german_transliterate.core import GermanTransliterate


_pad = "pad"
_eos = "eos"
_punctuation = "!'(),.? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Export all symbols:
ALL_SYMBOLS = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + [_eos]
)

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

def german_cleaners(text):
    """Pipeline for German text, including number and abbreviation expansion."""
    text = GermanTransliterate(replace={';': ',', ':': ' '}, sep_abbreviation=' -- ').transliterate(text)
    return text

class Processor():
    """German processor."""

    def __init__(self):
        self.symbol_to_id = {symbol: id for id, symbol in enumerate(ALL_SYMBOLS)}
        self.eos_id = self.symbol_to_id["eos"]

    def text_to_sequence(self, text):
        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self._symbols_to_sequence(
                    german_cleaners(text)
                )
                break
            sequence += self._symbols_to_sequence(
                german_cleaners(m.group(1))
            )
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        # add eos tokens
        sequence += [self.eos_id]
        return sequence

    def _symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]

    def _arpabet_to_sequence(self, text):
        return self._symbols_to_sequence(["@" + s for s in text.split()])

    def _should_keep_symbol(self, s):
        return s in self.symbol_to_id and s != "_" and s != "~"



def prepare_input(input_ids):
    return (tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
          tf.convert_to_tensor([len(input_ids)], tf.int32),
          tf.convert_to_tensor([0], dtype=tf.int32))

processor = Processor()

def infer_tflite(input_text, interpreter, mbmelgan_interpreter):
    input_ids = processor.text_to_sequence(input_text)
    interpreter.resize_tensor_input(input_details[0]['index'], [1, len(input_ids)])
    interpreter.allocate_tensors()
    input_data = prepare_input(input_ids)
    for i, detail in enumerate(input_details):
        interpreter.set_tensor(detail['index'], input_data[i])

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    mel_outputs = interpreter.get_tensor(output_details[0]['index'])
    mbmelgan_interpreter.resize_tensor_input(mbmelgan_input_details[0]['index'], mel_outputs.shape)
    mbmelgan_interpreter.allocate_tensors()
    mbmelgan_interpreter.set_tensor(mbmelgan_input_details[0]['index'], mel_outputs)
    mbmelgan_interpreter.invoke()

    # Get audio and remove noise at the end
    audio = mbmelgan_interpreter.get_tensor(mbmelgan_output_details[0]['index'])[0, :-1024, 0]
    return audio



if __name__ == "__main__":
    import tensorflow as tf
    path_to_melgan = tf.keras.utils.get_file(
        'german-tts-mbmelgan-lite.tar.gz',
        'https://storage.googleapis.com/mys-released-models/german-tts-mbmelgan-lite.tar.gz',
        extract=True,
        cache_subdir='german-tts-mbmelgan'
    )
    
    path_to_tacotron2 = tf.keras.utils.get_file(
        'german-tts-tacotron2-lite.tar.gz',
        'https://storage.googleapis.com/mys-released-models/german-tts-tacotron2-lite.tar.gz',
        extract=True,
        cache_subdir='german-tts-tacotron2'
    )

    # Load TFLite models and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path_to_tacotron2[:-6] + "tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    mbmelgan_interpreter = tf.lite.Interpreter(model_path=path_to_melgan[:-6] + "tflite")
    mbmelgan_interpreter.allocate_tensors()

    # Get input and output tensors.
    mbmelgan_input_details = mbmelgan_interpreter.get_input_details()
    mbmelgan_output_details = mbmelgan_interpreter.get_output_details()
    start = time.time()
    audio = infer_tflite("Möchtest du das meiner Frau erklären? Nein? Ich auch nicht.", interpreter, mbmelgan_interpreter)
    duration = time.time() - start
    print(F"it took {duration} secs")
    wavfile.write("sample_tflite.wav", 22050, audio)