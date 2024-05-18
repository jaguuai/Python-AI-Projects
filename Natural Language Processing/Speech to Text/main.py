from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource 
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# 1. Setup STT Service
apikey = 'your ibm cloud api'
url = 'your ibm cloud url'

# Setup stt Service
authenticator = IAMAuthenticator(apikey)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(url)

# open audio source and convert
with open('winston.mp3', 'rb') as f:
    res = stt.recognize(audio=f, content_type='audio/mp3', model='en-US_NarrowbandModel').get_result()

print(res)

text = res['results'][0]['alternatives'][0]['transcript']

with open('output.txt', 'w') as out:
   out.writelines(text)