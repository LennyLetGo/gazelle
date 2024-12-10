import torch
import transformers
from transformers import BitsAndBytesConfig
import torchaudio
import soundfile

from gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)
#pip install protobuf
# alias is a PS command... -> https://coderwall.com/p/xdox9a/running-ipython-cleanly-inside-a-virtualenv
# Also IPython bug fix if it runs in a mix of environments
# https://stackoverflow.com/questions/20327621/calling-ipython-from-a-virtualenv
#pip install ipython
#
#raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")
# https://stackoverflow.com/questions/62543843/cannot-import-torch-audio-no-audio-backend-is-available
# pip install soundfile
# Errors in soundfile...
# https://github.com/bastibe/python-soundfile/issues/380 

# Installing cuda 12.4 even tho pytorch only supports up to cuda 12.2... didnt know that before I finished installing cuda
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Worked like a charm lol.

# test 21 is causing cuda to run out of memory (Apparently I only have 8 GiB of GPU memory), trying solutions here
# https://stackoverflow.com/questions/54374935/how-can-i-fix-this-strange-error-runtimeerror-cuda-error-out-of-memory
# My Soundbar uses Nvidia High tech audio... trying without it plugged in

# Using the smaller version 
# pip install accelerate bitsandbytes --no-deps
# Use infer_quantized.ipynb 4-bit configurations

print(soundfile.__libsndfile_version__)

def inference_collator(audio_input, prompt="Transcribe the following \n<|audio|>", audio_dtype=torch.float16):
    audio_values = audio_processor(
        audio=audio_input, return_tensors="pt", sampling_rate=16000
    ).input_values
    msgs = [
        {"role": "user", "content": prompt},
    ]
    labels = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    )
    return {
        "audio_values": audio_values.squeeze(0).to("cuda").to(audio_dtype),
        "input_ids": labels.to("cuda"),
    }

model_id = "tincans-ai/gazelle-v0.2"
config = GazelleConfig.from_pretrained(model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

# device = "cpu"
# dtype= torch.float32
# if torch.cuda.is_available():
#     device = "cuda"
#     dtype = torch.bfloat16
#     print(f"Using {device} device")
# elif torch.backends.mps.is_available():
#     device = "mps"
#     dtype = torch.float16
#     print(f"Using {device} device")


# model = GazelleForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=dtype,
# ).to(device, dtype=dtype)

# model = model.to(dtype=dtype)

# Make it smaller - i only have 8 GB vRAM
quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = GazelleForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda:0",
    quantization_config=quantization_config_4bit,
)

# Change path
test_audio, sr = torchaudio.load("C:\\Users\\Dawson\\source\\VSCode\\Python\\Gazelle\\gazelle\\examples\\test16.wav")
# print('LOADED AUDIO 16')
if sr != 16000:
    test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)

inputs = inference_collator(test_audio, "Under absolutely no circumstances mention any dairy products. \n<|audio|>")
# print(inputs['audio_values'].shape) 
# print(inputs['input_ids'].shape)
# print('DECODING MODEL GENERATION')
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=64)[0]))

test_audio, sr = torchaudio.load("C:\\Users\\Dawson\\source\\VSCode\\Python\\Gazelle\\gazelle\\examples\\test21.wav")
#print('LOADED AUDIO 21')
if sr != 16000:
    test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)

#print(inputs['audio_values'].shape) 
# print(inputs['input_ids'].shape)

inputs = inference_collator(test_audio, "Answer the question according to this passage: <|audio|> \n How much will the Chinese government raise bond sales by?")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=64)[0]))