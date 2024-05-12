"""
GoaL:Generatic Poetic Text
Not:
1.Why text is lowercasing?
Lowercasing the text is a common preprocessing step in natural language processing tasks, including generating poetry or any other text. Here's why we lowercase the text:
Normalization: Lowercasing helps in normalizing the text. Since language is not case-sensitive in most cases, lowercase representation helps in treating the same words with different cases as identical. For example, "The" and "the" will be considered the same word after lowercasing.
Reducing Vocabulary Size: Lowercasing reduces the vocabulary size by collapsing words with different cases into a single representation. This simplifies the model training process and reduces the computational complexity.
Consistency: Lowercasing ensures consistency throughout the text data. By converting all characters to lowercase, we maintain uniformity in the text, which can help in improving the model's generalization performance.
Avoiding Redundancy: Lowercasing avoids redundancy in the vocabulary. Without lowercasing, the model may treat "Word" and "word" as different tokens, leading to redundant representations.
Overall, lowercasing the text is a standard practice to preprocess text data before feeding it into machine learning models, including AI applications such as generating poetry. It helps in improving model performance and simplifying the text processing pipeline.

"""
import random  # Python library used for generating random numbers.
import numpy as np  # Python library used for scientific calculations and data processing.
import tensorflow as tf  # Popular Python library for deep learning and artificial intelligence applications.
from tensorflow.keras.models import Sequential  # Module from the Keras library for creating sequential models.
from tensorflow.keras.layers import LSTM, Dense, Activation  # Keras modules containing artificial neural network layers and activation functions.
from tensorflow.keras.optimizers import RMSprop  # Optimization algorithm used for optimizing stochastic gradient descent (SGD).

filepath=tf.keras.utils.get_file("shakespeare.txt","https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

text=open(filepath,"rb").read().decode(encoding="utf-8").lower()
text=text[300000:800000]
characters=sorted(set(text))
char_to_index=dict((c,i)for i, c in enumerate(characters))
index_to_char=dict((i,c)for i, c in enumerate(characters))

SEQ_LENGTH=40
STEP_SIZE=3

sentences=[]
next_characters=[]

for i in range(0,len(text)-SEQ_LENGTH,STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])
    
x=np.zeros((len(sentences),SEQ_LENGTH,len(characters)),dtype=bool)
y=np.zeros((len(sentences),len(characters)),dtype=bool)

for i,sentence in enumerate(sentences):
    for t,character in enumerate(sentence):
        x[i,t,char_to_index[character]]=1
    y[i,char_to_index[next_characters[i]]]=1

model=Sequential()
model.add(LSTM(128,input_shape=(SEQ_LENGTH,len(characters))))
model.add(Dense(len(characters)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01))

model.fit(x,y,batch_size=250,epochs=4)

model.save("textgenerator.keras")

model = tf.keras.models.load_model("textgenerator.keras")

def sample(preds,temperature=0.1):
    preds=np.asarray(preds).astype("float64")
    preds=np.log(preds)/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1, preds,1)
    return np.argmax(probas)

def generate_text(length,temperature):
    start_index=random.randint(0,len(text)- SEQ_LENGTH-1)
    generated=""
    sentence=text[start_index: start_index + SEQ_LENGTH]
    generated +=sentence
    for i in range(length):
        x=np.zeros((len(sentences),SEQ_LENGTH,len(characters)))
        for t, character in enumerate(sentence):
            if t < SEQ_LENGTH: 
             x[0,t,char_to_index[character]]=1

        predictions=model.predict(x,verbose=0)[0]
        next_index=sample(predictions,temperature)
        next_character=index_to_char[next_index]
        
        generated +=next_character
        sentence= sentence[i:] + next_character
    return generated  

print("*********0.2**********")
print(generate_text(300,0.2))
print("*********0.4**********")
print(generate_text(300,0.4))
print("*********0.6**********")
print(generate_text(300,0.6))
print("*********0.8**********")
print(generate_text(300,0.8))
print("*********1**********")
print(generate_text(300,1.0))









