import re
import string
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics.pairwise import cosine_similarity
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

class TokenSimilarity:
    def load_pretrained(self, from_pretrained:str):
        self.tokenizer = BertTokenizer.from_pretrained(from_pretrained)
        self.model = TFBertModel.from_pretrained(from_pretrained)
        
    def __cleaning(self, text:str):
        # clear punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # clear multiple spaces
        text = re.sub(r'/s+', ' ', text).strip()

        return text
        
    def __process(self, first_token:str, second_token:str):
        inputs = self.tokenizer([first_token, second_token],
                                max_length=self.max_length,
                                truncation=self.truncation,
                                padding=self.padding,
                                return_tensors='tf')

        attention = inputs.attention_mask

        outputs = self.model(**inputs)

        # get the weights from the last layer as embeddings
        embeddings = outputs[0] # when used in older transformers version
        # embeddings = outputs.last_hidden_state # when used in newer one

        # add more dimension then expand tensor
        # to match embeddings shape by duplicating its values by rows
        mask = tf.expand_dims(attention, -1)
        mask = tf.cast(mask, tf.float32)
        mask = tf.broadcast_to(mask, tf.shape(embeddings))

        masked_embeddings = embeddings * mask
        
        # MEAN POOLING FOR 2ND DIMENSION
        # first, get sums by 2nd dimension
        # second, get counts of 2nd dimension
        # third, calculate the mean, i.e. sums/counts
        summed = tf.reduce_sum(masked_embeddings, axis=1)
        counts = tf.clip_by_value(tf.reduce_sum(mask, axis=1), clip_value_min=1e-9, clip_value_max=float('inf'))
        mean_pooled = summed/counts
        
        # return mean pooling as numpy array
        return mean_pooled.numpy()
        
    def predict(self, first_token:str, second_token:str,
                return_as_embeddings:bool=False, max_length:int=16,
                truncation:bool=True, padding:str="max_length"):
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

        first_token = self.__cleaning(first_token)
        second_token = self.__cleaning(second_token)

        mean_pooled_arr = self.__process(first_token, second_token)
        if return_as_embeddings:
            return mean_pooled_arr

        # calculate similarity
        similarity = cosine_similarity([mean_pooled_arr[0]], [mean_pooled_arr[1]])

        return similarity

@app.get("/calculate")
async def calculate():    
    model_name ='cahya/bert-base-indonesian-1.5G'
    model = TokenSimilarity()
    model.load_pretrained(model_name)
    
    query = 'pupuk npk'
    item = 'PUPUK NPK MUTIARA 16-16-16 ORIGINAL KEMASAN PABRIK 1KG'

    results = model.predict(query, item)
    
    results = np.array(results).tolist()

    return JSONResponse(content=results)
    
# @app.get("/search")
# async def search_items(search:str):
#     base_url = "https://pupukin-prod-l6hx3dk4bq-et.a.run.app/api"
#     endpoint = "/search/items"
#     url = base_url + endpoint

#     payload = {
#         "search": search
#     }

#     response = requests.post(url, json=payload)
#     return response.json()



@app.get("/")
async def index():
    return "LIVE!"


