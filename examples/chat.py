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

def establish_tcp_connection():
    pass

def send_data():
    pass

def receive_data():
    pass

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
    while(True):
        # use the microphone as source for input.
        with sr0.Microphone() as source2: 
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            r.adjust_for_ambient_noise(source2, duration=0.2)
                
            print('waiting on your prompt...')
            #listens for the user's input 
            audio2 = r.listen(source2)
            # add the audio to the conversation history
            # recognize speech using Google Speech Recognition
            try:
                # for testing purposes, we're just using the default API key
                # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                # instead of `r.recognize_google(audio)`
                conversation += f"USER:{r.recognize_google(audio2)}\n"
            except sr0.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr0.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

            # Sampling frequency
            freq = 44100
            f = open("C:\\Users\\Dawson\\source\\VSCode\\Python\\Gazelle\\gazelle\\examples\\user.wav", 'wb')
            f.write(audio2.get_wav_data())
            f.close()
            #write(, freq, )

            # lets try not saving the audio to a file first -> doesnt work save file then do th
            test_audio, sr = torchaudio.load("C:\\Users\\Dawson\\source\\VSCode\\Python\\Gazelle\\gazelle\\examples\\user.wav")
            if sr != 16000:
                test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)
            #context = 'You are a goblin under a bridge scrounging for goods, food, and collectables. You\'re hostile if provoked'
            context = f'''Once upon a time, in the lush woodlands of Eldoria, there lived a goblin named Griznak. Griznak was unlike other goblins; he possessed a curious mind and a gentle heart, traits rarely found among his kind. Born into a tribe known for its penchant for mischief and mayhem, Griznak often felt like an outsider, misunderstood and unappreciated.

Growing up under the shadow of a majestic stone bridge that spanned a crystal-clear stream, Griznak found solace in the natural beauty surrounding him. While his fellow goblins reveled in causing chaos for nearby villages and travelers, Griznak preferred to spend his days exploring the depths of the forest and observing the creatures that called it home.

Despite his peaceful nature, Griznak's tribe expected him to follow in their footsteps, wreaking havoc wherever he went. But Griznak couldn't bring himself to harm others or indulge in the senseless destruction his kin enjoyed. Instead, he sought refuge beneath the bridge, carving out a humble abode where he could escape the judgmental eyes of his fellow goblins.

Under the bridge, Griznak found comfort in solitude. He spent his days tending to a small garden of wildflowers and cultivating his knowledge of the forest's flora and fauna. He befriended the creatures that dwelled in the shadows, earning their trust through acts of kindness and compassion.

As time passed, Griznak's reputation as the "Bridge Goblin" grew among the villagers who traversed the stone structure above. Though they had heard tales of goblins as mischievous troublemakers, they soon discovered that Griznak was different. He would often leave small gifts of foraged berries or freshly picked flowers for those who passed by, earning him the affection and respect of the locals.

Despite his peaceful existence, Griznak knew that his bond with the villagers was fragile. Deep down, he feared that his true nature as a goblin would be revealed, shattering the fragile trust he had worked so hard to build. But for now, beneath the bridge, Griznak found a sense of belonging unlike any he had known before, surrounded by the beauty of the forest and the gentle murmur of the stream.'''
            # prompt = f'''
            # "Continue the conversation given the context and conversation history. Do not prefix your responses with '[name]:' 
            # Context: [{context}]
            # Conversation history: [{conversation}] 
            # Audio: \n<|audio|>"
            # '''
            
            # rules = f"""
            # Do not prefix your responses with '[name]:'
            # prefix your response using one of the following codes:
            # '<RESPOND>' = respond normally
            # '<BEGINATTACK> = when the user says they will attack Griznak, or when the user insults Grisnak
            # '<BEGINTRADE>' = Griznak agrees to trade
            # """
            rules = f"""
            Do not prefix your responses with '[name]:'
            """
            prompt = f'''
            "Pretend you're a goblin named Griznak. 
            Modify your response using the rules when the condition in the rules is met.  
            Context: [{context}]
            Conversation history: [{conversation}]
            Rules: [{rules}] 
            Audio: \n<|audio|>"
            '''
            
            #prompt = f"Continue the conversation given the context. Context [{context}] Audio: \n<|audio|>"
            
            # Try preprocessing the audio to determine anger or trade
            preprocessing_prompt = "If the recepient is in danger: Respond with '1', else: respond with '0'. \n<|audio|>"
            #preprocessing_prompt = f"""This is the conversation {conversation}. Only respond with the Rate of danger YOU faces on a scale of 0 to 10 \n<|audio|>"""
            inputs = inference_collator(test_audio, preprocessing_prompt)
            result = tokenizer.decode(model.generate(**inputs, max_new_tokens=2)[0])
            m = result.split('[/INST]')
            response = m[1].split('</s>')[0]
            print(f"Griznak in Danger? {response}")
            # Empty cache
            torch.cuda.empty_cache()
            
            inputs = inference_collator(test_audio, prompt)
            # run the model      
            result = tokenizer.decode(model.generate(**inputs, max_new_tokens=256)[0])

            m = result.split('[/INST]')
            response = m[1].split('</s>')[0]
            if(response.find(previousModelResponse) == -1):
                print(response)
                conversation += f"YOU: {response}\n"
                previousModelResponse = response
                try:  
                    # # Using google to recognize audio
                    # MyText = r.recognize_google(audio2)
                    # MyText = MyText.lower()

                    # let the goblin speak
                    engine.say(response) 
                    engine.runAndWait()
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
            else:
                print("*Monster resets*")
                try:  
                    # # Using google to recognize audio
                    # MyText = r.recognize_google(audio2)
                    # MyText = MyText.lower()

                    # let the goblin speak
                    engine.say("You get light headed as the sky gets bright...") 
                    engine.runAndWait()
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
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
            # Get new model... Do this conditionally maybe? 
            # del model
            # model = GazelleForConditionalGeneration.from_pretrained(
            #     model_id,
            #     device_map="cuda:0",
            #     quantization_config=quantization_config_4bit,
            # )
        
