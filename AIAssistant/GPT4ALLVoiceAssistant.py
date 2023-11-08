import os
import time
import warnings
import pyttsx3
import speech_recognition as sr
import whisper
from gpt4all import GPT4All
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='whisper.transcribe', lineno=114)

# Configuration
WAKE_WORD = 'computer'
GPT_MODEL_PATH = '~/Local/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin'
TINY_MODEL_PATH = '~/.cache/whisper/tiny.pt'
BASE_MODEL_PATH = '~/.cache/whisper/base.pt'
OUTPUT_FILE = 'output.txt'

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_for_wake_word(audio):
    try:
        with open("wake_detect.wav", "wb") as f:
            f.write(audio.get_wav_data())
        result = whisper.load_model(TINY_MODEL_PATH).transcribe('wake_detect.wav')
        text_input = result['text']
        if WAKE_WORD in text_input.lower().strip():
            print("Wake word detected. Please speak your prompt to GPT4All.")
            speak('Listening')
            return True
    except Exception as e:
        print("Error detecting wake word:", e)
    return False

def prompt_gpt(audio):
    try:
        with open("prompt.wav", "wb") as f:
            f.write(audio.get_wav_data())
        result = whisper.load_model(BASE_MODEL_PATH).transcribe('prompt.wav')
        prompt_text = result['text']
        
        # Check if the user said 'stop listening' to exit the loop
        if 'stop listening' in prompt_text.lower():
            print('Stopping listening.')
            exit()
        if not prompt_text.strip():
            print("Empty prompt. Please speak again.")
            speak("Empty prompt. Please speak again.")
        else:
            print('User:', prompt_text)
            gpt_model = GPT4All(GPT_MODEL_PATH, allow_download=False)
            output = gpt_model.generate(prompt_text, max_tokens=200)
            print('GPT4All:', output)
            speak(output)
    except Exception as e:
        print("Prompt error:", e)

def main():
    source = sr.Microphone()
    r = sr.Recognizer()

    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    listening_for_wake_word = True
    print('\nSay', WAKE_WORD, 'to wake me up.\n')

    while True:
        with sr.Microphone() as source:
            audio = r.listen(source)

        if listening_for_wake_word:
            if listen_for_wake_word(audio):
                listening_for_wake_word = False
        else:
            prompt_gpt(audio)

if __name__ == '__main__':
    main()