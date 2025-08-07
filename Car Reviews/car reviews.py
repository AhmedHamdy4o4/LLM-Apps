# laod libararies
from transformers import logging, pipeline
import os
import pandas as pd
from transformers import pipeline
from evaluate import load 
from sklearn.metrics import f1_score,accuracy_score

#load data 
logging.set_verbosity(logging.WARNING) 
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1' 
# Load the car reviews dataset
df = pd.read_csv('data/car_reviews.csv', sep=';')
print(df.info()) 

# Task 1 : perform sentiment model 
model = pipeline(task = "sentiment-analysis",model = "destil-bert-base-finetuned-sst-2-english")
# get output from sentiment analysis model as prediction
df["pred"] = [model(df['Review'].tolist())[i]['label'] for i in range(df.shape[0])]

# encode model output and class labels 0's and 1's to feed to accuracy model 
replace_pred = {"POSITIVE":1,"NEGATIVE":0}
df['pred'] = df['pred'].replace(replace_pred)
df['Class'] = df['Class'].replace(replace_pred)
df.head()

# perform accuracy metrices [f1 score, accuracy]
f1_result = f1_score(df.Class,df.pred)
accuracy_result = accuracy_score(df.Class,df.pred)
print(f"F1 result : {f1_result}\nAccuracy result : {accuracy_result}")

# Task2: English-to-Spanish translation
# data acquisition
data = []
with open('data/reference_translations.txt','r') as file :
    data += file.readlines()
data = [d.strip() for d in data]

# instanciate translation model 
translator_model = pipeline(
    task="translation_es_to_en",
    model="Helsinki-NLP/opus-mt-es-en")
translated_review = translator_model(data)
# unpack model output 
translated_review = [translated_review[i]['translation_text'] for i  in range(len(data))]

# evalute the model using bleu score 
bleu = load("bleu")
references = [[ref] for ref in data]
bleu_score = bleu.compute(predictions= translated_review,references = references)
print(f"bleu score : {bleu_score['bleu']}")

# Task 3 QA analysis 
#instanciate Question-Answering model & feed it with question and context(base knowledge)
QA_model = pipeline('question-answering',model="deepset/minilm-uncased-squad2")
q = "What did he like about the brand?"
answer = QA_model(context = df.Review[0],question=q)['answer']
answer
print(f"question : {q} \nAnswer : {answer}")

# Task 4 summarization 
# instanciate the model and feed it with data
summarizer = pipeline(task = "summarization", model = "sshleifer/distilbart-cnn-12-6")
summarized_text = summarizer(df.Review[0],min_length=50,max_length=55,do_sample=False)[0]['summary_text']
print(f"Review: {df.Review[0]}\n\n\nsummaried review : {summarized_text}")

# evaluate summarization model by tocixity and ragard (bais) metrics 
toxicity = load("toxicity")
regard = load("regard")
toxicity_value = toxicity.compute(references = [df.Review[0]],predictions=[summarized_text])
regard_value = regard.compute(data=summarized_text)
print(f"Text : {df.Review[0]}\n\nToxicity value : {toxicity_value}\nRegard output : {regard_value['regard'][0][0]['label']}")