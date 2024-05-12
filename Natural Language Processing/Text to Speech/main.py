# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:59:14 2024

@author: Text to speech
"""
# authenticate
api_key="your ıbm key"
url="your ibm cloud url"


from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator=IAMAuthenticator(api_key)

tts=TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(url)

# convert a string
with open("./speech.mp3", "wb") as audio_file:
    res = tts.synthesize("Hello World", accept="audio/mp3", voice="en-US_AllisonV3Voice", timeout=60).get_result()
    audio_file.write(res.content)
    if res.status_code != 200:
        print("Error occurred while synthesizing speech:", res.text)
    else:
        audio_file.write(res.content)
# convert from a file
with open('churchill.txt', 'r') as f:
    text = f.readlines()
text = [line.replace('\n','') for line in text]
text = ''.join(str(line) for line in text)
with open('./winston.mp3', 'wb') as audio_file:
    res = tts.synthesize(text, accept='audio/mp3', voice='en-GB_JamesV3Voice').get_result()
    audio_file.write(res.content) 
    
# using new language models
voice="de-DE_BirgitV3Voice"
german_text="Wir werden bis zum Ende weitermachen, wir werden in Frankreich kämpfen, wir werden auf den Meeren und Ozeanen kämpfen"
with open('./germanspeech.mp3', 'wb') as audio_file:
    res = tts.synthesize(german_text, accept='audio/mp3', voice='en-GB_JamesV3Voice').get_result()
    audio_file.write(res.content) 
















