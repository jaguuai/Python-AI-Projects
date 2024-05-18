# 1-Authenticate
apikey = 'YOUR TTS APIKEY HERE'
url = 'YOUR TTS URL HERE'
# import deps
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
# Setup service
authenticator = IAMAuthenticator(apikey)
lt = LanguageTranslatorV3(version='2023-05-10', authenticator=authenticator)
lt.set_service_url(url)

# 2. Translate
translation = lt.translate(text='We are sinking.', model_id='en-de').get_result()

result_trans=translation['translations'][0]['translation']
print(result_trans)

# 3. Identify Languages

language = lt.identify('Wir sinken.').get_result()
print(language)

# AI Travel Guide -Text to Speech
ttsapikey = 'YOUR TTS APIKEY HERE'
ttsurl = 'YOUR TTS URL HERE'
from ibm_watson import TextToSpeechV1
# Authenticate
ttsauthenticator = IAMAuthenticator(ttsapikey)
tts = TextToSpeechV1(authenticator=ttsauthenticator)
tts.set_service_url(ttsurl)

translation = lt.translate(text='We are sinking! Please send help!', model_id='en-zh').get_result()
text = translation['translations'][0]['translation']

with open('./help.mp3', 'wb') as audio_file:
    res = tts.synthesize(text, accept='audio/mp3', voice='zh-CN_LiNaVoice').get_result()
    audio_file.write(res.content)