
# coding: utf-8

# # Imports and classes

# In[26]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import re
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from datetime import datetime
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/chrisss/Downloads/NLP with Disaster Tweets-942b991d9d77.json"
from sklearn.model_selection import train_test_split
import nltk
from spellchecker import SpellChecker
#cant install pyahocorasick so some reason 
#from contractions import CONTRACTION_MAP
from bs4 import BeautifulSoup
from google.cloud import storage
from google.cloud import automl_v1beta1 as automl
import string


#from automlwrapper import AutoMLWrapper


# This notebook utilizes a utility script that wraps much of the AutoML Python client library, to make the code in this notebook easier to read. Feel free to check out the utility for all the details on how we are calling the underlying AutoML Client Library!

# In[2]:


# Values used for automl
PROJECT_ID = 'nlp-with-disaster-tweets'
BUCKET_NAME = 'nlp-with-disaster-tweets-lcm'

BUCKET_REGION = 'us-central1' # Region must be us-central1




storage_client = storage.Client(project=PROJECT_ID)
tables_gcs_client = automl.GcsClient(client=storage_client, bucket_name=BUCKET_NAME)
automl_client = automl.AutoMlClient()
# Note: AutoML Tables currently is only eligible for region us-central1. 
prediction_client = automl.PredictionServiceClient()
# Note: This line runs unsuccessfully without each one of these parameters
tables_client = automl.TablesClient(project=PROJECT_ID, region=BUCKET_REGION, client=automl_client, gcs_client=tables_gcs_client, prediction_client=prediction_client)


# In[3]:


# Create your GCS Bucket with your specified name and region (if it doesn't already exist)
bucket = storage.Bucket(storage_client, name=BUCKET_NAME)
if not bucket.exists():
    bucket.create(location=BUCKET_REGION)


# ### GCS upload/download utilities
# These functions to  upload and download of files from the kernel to Google Cloud Storage3

# In[4]:


#move local files to GCS,

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    
def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(full_file_path)


# In[11]:


nlp_train_df = pd.read_csv('trainTrans.csv', encoding='latin-1')
nlp_test_df = pd.read_csv('testTrans.csv', encoding='latin-1')
def callback(operation_future):
    result = operation_future.result()


# In[12]:


nlp_train_df.tail()


# In[7]:


nlp_train_df.tail()


# ### Data spelunking
# How often does 'fire' come up in this dataset?

# # Data cleaning
#     Remove HTML
#     Tokenization + Remove punctuation
#     Remove stop words
#     Lemmatization or Stemming

# In[5]:


tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
#cant install pyahocorasick so some reason 
"""def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

"""
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text





def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)


spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)




# In[5]:


x=0
while(len(nlp_train_df)>x):
    print(x)
    
    nlp_train_df['text'][x] = strip_html_tags(nlp_train_df['text'][x])
    nlp_train_df['text'][x] = remove_accented_chars(nlp_train_df['text'][x])
    nlp_train_df['text'][x] = remove_special_characters(nlp_train_df['text'][x])
    nlp_train_df['text'][x] = simple_stemmer(nlp_train_df['text'][x])
    nlp_train_df['text'][x] = remove_stopwords(nlp_train_df['text'][x])
    nlp_train_df['text'][x] = correct_spellings(nlp_train_df['text'][x])
    nlp_train_df['text'][x] = remove_punct(nlp_train_df['text'][x])


    x=x+1


# In[6]:


x=0
while(len(nlp_test_df)>x):
    print(x)
    
    nlp_test_df['text'][x] = strip_html_tags(nlp_test_df['text'][x])
    nlp_test_df['text'][x] = remove_accented_chars(nlp_test_df['text'][x])
    nlp_test_df['text'][x] = remove_special_characters(nlp_test_df['text'][x])
    nlp_test_df['text'][x] = simple_stemmer(nlp_test_df['text'][x])
    nlp_test_df['text'][x] = remove_stopwords(nlp_test_df['text'][x])
    nlp_test_df['text'][x] = correct_spellings(nlp_test_df['text'][x])
    nlp_test_df['text'][x] = remove_punct(nlp_test_df['text'][x])


    x=x+1


# In[7]:


nlp_test_df.to_csv('testTrans.csv', index=False, header=False) 


# In[6]:


nlp_train_df.to_csv('trainTrans.csv', index=False, header=False) 


# In[13]:


nlp_train_df.loc[nlp_train_df['text'].str.contains('fire', na=False, case=False)]


# Does the presence of the word 'fire' help determine whether the tweets here are real or false?

# In[14]:


nlp_train_df.loc[nlp_train_df['text'].str.contains('fire', na=False, case=False)].target.value_counts()


# In[15]:


#upload transformed data to google cloud, make sure you save changes into the csv file.

upload_blob(BUCKET_NAME, 'trainTrans.csv', 'trainTrans.csv')
upload_blob(BUCKET_NAME, 'testTrans.csv', 'testTrans.csv')




# ## Create (or retreive) dataset
# Check to see if this dataset already exists. If not, create it

# In[16]:


# create a dataset within AutoML tables that references your saved data in GCS.
model_display_name = 'kaggle_starter_model1'
dataset_display_name = 'kaggle_tweets'
new_dataset = False
try:
    dataset = tables_client.get_dataset(dataset_display_name=dataset_display_name)
except:
    new_dataset = True
    dataset = tables_client.create_dataset(dataset_display_name)


# In[17]:


bucket = storage.Bucket(storage_client, name=BUCKET_NAME)
if not bucket.exists():
    bucket.create(location=region)


# In[18]:


# gcs_input_uris have the familiar path of gs://BUCKETNAME//file
#give it the path to where the relevant data is in GCS (GCS file paths follow the format gs://BUCKET_NAME/file_path) and import your data.
if new_dataset:
    gcs_input_uris = ['gs://' + BUCKET_NAME + '/train.csv']

    import_data_operation = tables_client.import_data(
        dataset=dataset,
        gcs_input_uris=gcs_input_uris
    )
    print('Dataset import operation: {}'.format(import_data_operation))

    # Synchronous check of operation status. Wait until import is done.
    import_data_operation.result()
print(dataset)


# ### Export to CSV and upload to GCS

# In[10]:


# save transformed data to csv
#nlp_train_df[['text','target']].to_csv('train.csv', index=False, header=False) 


# In[23]:


model_display_name = 'tutorial_Newmodel'
TARGET_COLUMN = 'target'
ID_COLUMN = 'id'

# TODO: File bug: if you run this right after the last step, when data import isn't complete, you get a list index out of range
# There might be a more general issue, if you provide invalid display names, etc.

tables_client.set_target_column(
    dataset=dataset,
    column_spec_display_name=TARGET_COLUMN
)


# In[24]:


# Make all columns nullable (except the Target and ID Column)
for col in tables_client.list_column_specs(PROJECT_ID,BUCKET_REGION,dataset.name):
    if TARGET_COLUMN in col.display_name or ID_COLUMN in col.display_name:
        continue
    tables_client.update_column_spec(PROJECT_ID,
                                     BUCKET_REGION,
                                     dataset.name,
                                     column_spec_display_name=col.display_name,
                                     type_code=col.data_type.type_code,
                                     nullable=True)



# In[21]:


nlp_train_df[['id','text','target']].head()


# ## Kick off the training for the model
# And retrieve the training info after completion. 
# Start model deployment.

# In[25]:


# Train the model. This will take hours (up to your budget). AutoML will early stop if it finds an optimal solution before your budget.

#1 hour
TRAIN_BUDGET = 1000 # (specified in milli-hours, from 1000-72000)
model = None
try:
    model = tables_client.get_model(model_display_name=model_display_name)
except:
    response = tables_client.create_model(
        model_display_name,
        dataset=dataset,
        train_budget_milli_node_hours=TRAIN_BUDGET,
        exclude_column_spec_names=[ID_COLUMN, TARGET_COLUMN]
    )
    print('Create model operation: {}'.format(response.operation))
    # Wait until model training is done.
    model = response.result()
# print(model)


# In[35]:


dataset


# # Batch Predict on your Test Dataset
# 
# point our newly created autoML model to our test file and spit out some new predictions.

# In[27]:


gcs_input_uris = 'gs://' + BUCKET_NAME + '/test.csv'
gcs_output_uri_prefix = 'gs://' + BUCKET_NAME + '/predictions'

batch_predict_response = tables_client.batch_predict(
    model=model, 
    gcs_input_uris=gcs_input_uris,
    gcs_output_uri_prefix=gcs_output_uri_prefix,
)
print('Batch prediction operation: {}'.format(batch_predict_response.operation))
# Wait until batch prediction is done.
batch_predict_result = batch_predict_response.result()
batch_predict_response.metadata


# ## Prediction
# Note that prediction will not run until deployment finishes, which takes a bit of time.
# However, once you have your model deployed, this notebook won't re-train the model, thanks to the various safeguards put in place. Instead, it will take the existing (trained) model and make predictions and generate the submission file.

# In[28]:


# The output directory for the prediction results exists under the response metadata for the batch_predict operation
# Specifically, under metadata --> batch_predict_details --> output_info --> gcs_output_directory
# Then, you can remove the first part of the output path that contains the GCS Bucket information to get your desired directory
gcs_output_folder = batch_predict_response.metadata.batch_predict_details.output_info.gcs_output_directory.replace('gs://' + BUCKET_NAME + '/','')
download_to_kaggle(BUCKET_NAME,'/kaggle/working','tables_2.csv', prefix=gcs_output_folder)



# In[29]:


preds_df = pd.read_csv("predictions.csv")
preds_df["class"] = 0

#submission_df.to_csv('submission.csv', index=False)


# In[30]:


#convert probablity to  either 0 or 1, where 0 is not a real disasters, 
x=0
while(len(preds_df)>x):
    
    if(preds_df['target_1_score'][x]>0.5):
        preds_df['class'][x] = 1

    x=x+1    


# In[31]:


preds_df


# ## Create submission output

# In[ ]:


preds_df.head()


# In[ ]:


#submission_df = pd.concat([nlp_test_df['id'], predictions_df['class']], axis=1)
#submission_df.head()


# In[ ]:


# predictions_df['class'].iloc[:10]
# nlp_test_df['id']


# In[ ]:


#submission_df = submission_df.rename(columns={'class':'target'})
#submission_df.head()


# ## Submit predictions to the competition!

# In[34]:


preds_df[['id','class']].sort_values(by=['id']).to_csv("submissions.csv", index=False, header=True)


# In[33]:


get_ipython().system(' ls -l submission.csv')

