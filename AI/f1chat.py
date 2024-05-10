#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#######################################################
#  Initialise Task D
#######################################################

import langcodes
import os
import requests
import uuid
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


# Cognitive Services API credentials
cog_key = '326b58bb68914fe9bcf6e922ac73bc56'
cog_endpoint = 'https://cognitivedhaval.cognitiveservices.azure.com/'
cog_region = 'uksouth'
text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint, credential=AzureKeyCredential(cog_key))


#Function that gets the language name and returns the language code
def get_language_code(language_name):
    try:
        language_code = langcodes.Language.find(language_name).language
    except ValueError:
        raise ValueError("Invalid language name")
    return language_code

    


# Function that reads printed text from an image using the Computer Vision API, returns the text from the image and the language that it is in
def read_text_from_image(image_path):
    
    computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))
   
    image_stream = open(image_path, "rb")
    
    read_results = computervision_client.recognize_printed_text_in_stream(image_stream)
    text = ''
    for region in read_results.regions:
        for line in region.lines:
            for word in line.words:
                text += word.text + ' '
            text += '\n'
    
    # Detect the language of the text
    response = text_analytics_client.detect_language([text])
    detected_language = response[0].primary_language.iso6391_name
    
    return (text, detected_language)


# Function that translates text from one language to another using the Text Translation API
def translate_text(text, from_lang='en', to_lang='en'):
    to_lang = get_language_code(to_lang)
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region': cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{
        'text': text
    }]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]




#######################################################
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring



#######################################################
#  Initialise Knowledgebase. 
#######################################################
import pandas
kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

#Function that checks the integrity of the kb
def kb_integrity(expr,kb):
    not_answer = Expression.negate(expr)
    answer = ResolutionProver().prove(not_answer,kb, verbose=False)
    if not answer:
        return True
    else:
        return False

#######################################################
#  Initialise AIML agent
#######################################################
import aiml
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

#Reads the Q/A pairs file
with open('f1QA.csv','r',encoding='utf-8') as QA:
    reader = csv.reader(QA)
    f1QA = list(reader)

#process the userinput
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)



vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
tfidf_vectors = vectorizer.fit_transform([q[0] for q in f1QA])


# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

#######################################################
# Welcome user
#######################################################
print("Hello and welcome to this F1 chat bot. You can ask me any question about Formula 1 racing.")

#For Task D
ext = ['.png', 'jpg','.jpeg']

#######################################################
# Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    
    #Essentially split the file extension from the input (TASK D)
    FE = ''
    for e in ext:
        if e in userInput:
            userInput = userInput.replace(e,"")
            FE = e
            break

    #pre-process user input and determine response agent
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
        
        #If nothing found from the AIML file it finds most similar question from QA pairs
        if not answer or answer.startswith('#99$'):
            user_vector = vectorizer.transform([preprocess_text(userInput)])
            similarities = cosine_similarity(user_vector, tfidf_vectors)[0]
            most_similar = similarities.argmax()
            if similarities[most_similar] > 0.65:
                answer = f1QA[most_similar][1]
            else:
                answer = "Sorry, please ask another question as I don't know this."
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break

        
        #TASK B
        # Here are the processing of the new logical component:
        elif cmd == 31: # if input pattern is "I know that * is a *"
            object,subject=params[1].split(' is a ')
            inputtedobject = object
            object = "_".join(object.split())
            object = object.lower()
            expr=read_expr(subject + '(' + object + ')')
            
            #check if the user is trying to enter a contradiction
            if kb_integrity(expr,kb): 
                kb.append(expr) 
                print('OK, I will remember that',inputtedobject,'is a', subject)
            else:
                print("A contradiction is found.\n")
        
        elif cmd == 32: # if the input pattern is "check that * is a *"
            object,subject=params[1].split(' is a ')
            object = "_".join(object.split())
            object = object.lower()
            expr=read_expr(subject + '(' + object + ')')
            answer=ResolutionProver().prove(expr, kb, verbose=False)
            if answer is not None:
                if answer:
                    print("This is correct.")
                else: 
                    not_expr = read_expr('not ' + subject + '(' + object + ')')
                    not_answer = ResolutionProver().prove(not_expr, kb, verbose=False)
                    if not_answer is not None and not_answer:
                        print("This is not true.")
                    else:
                        print("Sorry I don't know")
            else:
                print("error, try again...")
                
                    
        #Processing of Task D component
        elif cmd == 45:
            # if input pattern is "show text from * in *"
            parts = userInput.split()
            image_file = parts[3] + FE
            language = None
            if len(parts) > 4 and parts[4] == 'in' and len(parts) > 5:
                language = parts[5]

            # Read text from the image and detect the language
            image_path = os.path.join('images',image_file)
            if not os.path.exists(image_path):
                print("This file was not found, please try again.")

            
            else:
                text, detected_language = read_text_from_image(image_path)
                # Translate the text if requested
                if language:
                    translated_text = translate_text(text, from_lang=detected_language, to_lang=language)
                    print(translated_text)
                else:
                    print(text)
                
                
        
             
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer,"\n")
        
        
        
        