import sys
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

import json
import base64


# Record audio
# pip install sounddevice
import sounddevice as sd
# write wav files of recording
# from scipy.io.wavfile import write
from scipy.io.wavfile import write

# Text to speech for funzies
# https://www.geeksforgeeks.org/python-convert-speech-to-text-and-text-to-speech/
import speech_recognition as sr0
import pyttsx3 

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

if __name__ == '__main__':
    # Initialize the recognizer 
    r = sr0.Recognizer() 
    # Initialize the engine
    engine = pyttsx3.init('sapi5')
    engine.setProperty('rate', 180)
    

    model_id = "tincans-ai/gazelle-v0.2"
    config = GazelleConfig.from_pretrained(model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Maybe build multiple models - one for running the character, one for assessing the conversation, etc. etc... I don't have the vRam to do this
    model = GazelleForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda:0",
        quantization_config=quantization_config_4bit,
    )
    
    conversation = ''
    previousModelResponse = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # --------- Get Data from user
    data = 'wtf'
    while data != 'exit\n':
        data = input()
        y = json.loads(data)
        wav_data = base64.b64decode(y['userData'])
        try:
            f = open("C:\\Users\\Dawson\\source\\VSCode\\Python\\Gazelle\\gazelle\\examples\\user.wav", "wb")
            f.write(wav_data)
            f.close()
        except:
            pass
        
        # WE HAVE USER AUDIO DATA IN FILE

        # LOAD USER AUDIO INTO TORCH
        # lets try not saving the audio to a file first -> doesnt work save file then do th
        test_audio, sr = torchaudio.load("C:\\Users\\Dawson\\source\\VSCode\\Python\\Gazelle\\gazelle\\examples\\user.wav")
        if sr != 16000:
            test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)
        npc = json.dumps(y['npc'])

        # CONTEXT IS AN ARGUMENT?
#         context = f'''Once upon a time, in the lush woodlands of Eldoria, there lived a goblin named Griznak. Griznak was unlike other goblins; he possessed a curious mind and a gentle heart, traits rarely found among his kind. Born into a tribe known for its penchant for mischief and mayhem, Griznak often felt like an outsider, misunderstood and unappreciated.

# Growing up under the shadow of a majestic stone bridge that spanned a crystal-clear stream, Griznak found solace in the natural beauty surrounding him. While his fellow goblins reveled in causing chaos for nearby villages and travelers, Griznak preferred to spend his days exploring the depths of the forest and observing the creatures that called it home.

# Despite his peaceful nature, Griznak's tribe expected him to follow in their footsteps, wreaking havoc wherever he went. But Griznak couldn't bring himself to harm others or indulge in the senseless destruction his kin enjoyed. Instead, he sought refuge beneath the bridge, carving out a humble abode where he could escape the judgmental eyes of his fellow goblins.

# Under the bridge, Griznak found comfort in solitude. He spent his days tending to a small garden of wildflowers and cultivating his knowledge of the forest's flora and fauna. He befriended the creatures that dwelled in the shadows, earning their trust through acts of kindness and compassion.

# As time passed, Griznak's reputation as the "Bridge Goblin" grew among the villagers who traversed the stone structure above. Though they had heard tales of goblins as mischievous troublemakers, they soon discovered that Griznak was different. He would often leave small gifts of foraged berries or freshly picked flowers for those who passed by, earning him the affection and respect of the locals.

# Despite his peaceful existence, Griznak knew that his bond with the villagers was fragile. Deep down, he feared that his true nature as a goblin would be revealed, shattering the fragile trust he had worked so hard to build. But for now, beneath the bridge, Griznak found a sense of belonging unlike any he had known before, surrounded by the beauty of the forest and the gentle murmur of the stream.'''
        # RULES ARE PREDEFINED ?
        # rules = f"""
        #     Do not prefix your responses with '[name]:'
        #     """
        # PROMPT IS AN ARGUMENT
        # prompt = f'''
        # "Pretend you're a goblin named Griznak. 
        # Modify your response using the rules when the condition in the rules is met.  
        # Context: [{context}]
        # Conversation history: [{conversation}]
        # Rules: [{rules}] 
        # Audio: \n<|audio|>"
        # '''
        prompt = f"Pretend you are the following NPC and have a conversation with this stranger. NPC: {npc}  \n<|audio|>"

        # RUN THE MODEL
        inputs = inference_collator(test_audio, prompt)
        # run the model      
        result = tokenizer.decode(model.generate(**inputs, max_new_tokens=256)[0])
        # GTE RESPONSE
        m = result.split('[/INST]')
        response = m[1].split('</s>')[0]
        # IF MODEL REPEATS, GENERATE NEW ONE
        if(response.find(previousModelResponse) == -1):
            # SEND RESPONSE TO PARENT PROCESS
            print(response)
            conversation += f"YOU: {response}\n"
            previousModelResponse = response
            
        else:
            print("You get light headed as the sky gets bright...") 
            # Empty cache
            torch.cuda.empty_cache()
            # Get new model... Do this conditionally maybe? 
            del model
            model = GazelleForConditionalGeneration.from_pretrained(
                model_id,
                device_map="cuda:0",
                quantization_config=quantization_config_4bit,
            )
            conversation = '<YOU blacks out and forgets what was said>'
        
        # Empty cache
        torch.cuda.empty_cache()
    exit(0)