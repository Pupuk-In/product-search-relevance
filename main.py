import re
import string
from typing import Union
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    relevance: Union[float, None] = None   
     
class Input(BaseModel):
    items: list[Item]
    query: str
  
class MyModel:
    def load_pretrained(self, from_pretrained:str):
        self.tokenizer = BertTokenizer.from_pretrained(from_pretrained)
        self.model = TFBertModel.from_pretrained(from_pretrained)
        
    def __cleaning(self, text:str):
        text = text.translate(str.maketrans('', '', string.punctuation))
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

        embeddings = outputs[0] 
        mask = tf.expand_dims(attention, -1)
        mask = tf.cast(mask, tf.float32)
        mask = tf.broadcast_to(mask, tf.shape(embeddings))

        masked_embeddings = embeddings * mask
        
        summed = tf.reduce_sum(masked_embeddings, axis=1)
        counts = tf.clip_by_value(tf.reduce_sum(mask, axis=1), clip_value_min=1e-9, clip_value_max=float('inf'))
        mean_pooled = summed/counts
        
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

        similarity = cosine_similarity([mean_pooled_arr[0]], [mean_pooled_arr[1]])
        
        return similarity

model_name ='cahya/bert-base-indonesian-1.5G'
model = MyModel()
model.load_pretrained(model_name)

@app.post("/calculate/", response_model=list[Item])
async def calculate(input: Input):
    
    query = input.query
    items = input.items
    
    results = []
    
    for item in items:
        result = Item(id=item.id, name=item.name)
        result.relevance = model.predict(query, item.name)
        results.append(result)
          
    return results

@app.get("/")
async def index():
    return "LIVE!"


