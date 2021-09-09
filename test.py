import pandas as pd
import io

from textblob import TextBlob as tb
f = open('iHumanQuestions_wOptions_20210419.csv',"rb")
df2 = pd.read_csv(f)
possible_responses = df2["Statement.RESPONSE"].values.tolist()
paired_questions = df2["Statement.QUESTION"].values.tolist()

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np


from sentence_transformers import SentenceTransformer
lsbert_model = SentenceTransformer('bert-large-nli-mean-tokens')

import pyaudio
from rev_ai.models import MediaConfig
from rev_ai.streamingclient import RevAiStreamingClient
from six.moves import queue


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

# Sampling rate of your microphone and desired chunk size
rate = 44100
chunk = int(rate/10)

# Insert your access token here
access_token = "02dE4JynMyMV6NtH2NRPiyaURlQiAfpCS3o_TxhSbT_X9JicRl5EaGeZbFBjOTfJxeLz6CFY53_eeT5wJbAB0Ct9h6awA"

# Creates a media config with the settings set for a raw microphone input
example_mc = MediaConfig('audio/x-raw', 'interleaved', 44100, 'S16LE', 1)

streamclient = RevAiStreamingClient(access_token, example_mc)
import json
#cosine similarity equation
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

#read in the embedded paired questions from the modified Tom Bradford dataset

import io
import pickle
with open('lsEmbedded_pairedQuestions_20210419.csv', 'rb') as filehandle:
  ls_embedded_paired_questions = pickle.load(filehandle)

#bodyparts list for conversation linearity. list could be a dictionary and convert words to the preferred TB terms
body_parts_list = ["stomach", "belly", "abdominals", "abs", "heart", "head", "shoulders", "knees", "toes", "eye", "ear", "mouth", "nose", "chest", "leg", "lung", "lungs", "kidney", "kidneys"]

#function get_bodyPart
def get_bodyPart(text):
  ''' 
  This function checks if the input text contains a body part,
  and saves the last body part referenced in the text to the global variable "bodyPart".
  Run this function on the last question, then the last answer continuously throughout the conversation, 
  so the "bodyPart" variable is constantly updated with the last referenced body part in the conversation. 
  Parameters:
      :text: str
  Returns:
      :None, but if the text contained a body part, then it updates the global variable "bodyPart" to the last referenced body part in the text.
  '''
  import nltk
  tkn_text = nltk.word_tokenize(text)

  for word in tkn_text:
    if word in body_parts_list:
      global bodyPart
      bodyPart = word

#Function get_response
def get_response(text):
  '''
  This function finds the most similar question from the Tom Bradford + team modified dataset,
  and returns a dictionary containing the most similar question from the dataset, the similarity score, and the paired response.
  Also adds the response to the end of the list of patient responses, variable "patient_dialogue".
  Parameters:
      :text: str
  Returns:
      A dictionary containing the most similar question from the dataset, the similarity score, and the paired response.
  '''
  add_bodypart = False
  if not any(part in text for part in body_parts_list):
    if 'pain' in text:
      if bodyPart != 'none':  
        text = text + ' ' + bodyPart
        add_bodypart = True  

  text_vec = lsbert_model.encode([text])[0]

  similarities = []
  for i in range(len(possible_responses)):
    similarities.append(cosine(text_vec, ls_embedded_paired_questions[i]))
  
  max_sim_index = max((v, i) for i, v in enumerate(similarities))[1]
   
  response = possible_responses[max_sim_index]

  paired_q = paired_questions[max_sim_index]
  similarity = similarities[max_sim_index]

  if add_bodypart == True:
    global add_bodypart_count
    if add_bodypart_count == 0:
      response = "Did you mean the " + bodyPart + " pain? " + response
      add_bodypart_count += 1
    elif add_bodypart_count == 1:
      response = "The " + bodyPart + " pain? " + response 
      add_bodypart_count += 1
    elif add_bodypart_count == 2:
      response = "Oh, you mean the " + bodyPart + " pain? " + response
      add_bodypart_count == 0

  dict_results = {}
  dict_results['response'] = response
  dict_results['paired_q'] = paired_q
  dict_results['similarity'] = similarity

  return(dict_results)
#end get_response function

#Function seperate_question
def seperate_question(text):
  '''
  This function checks if the input text has a "non-question half" and a "question half",
  and if so, it returns only the "question half".
  Parameters:
      :text: str
  Returns:
      :All or just the "question half" of the input text
  '''
  nonQuestionPunctuation = ['.', ';', '!']
  qPunct_i = []
  nonQpunct_i = []
  for i in range(len(text)):
    if text[i] in nonQuestionPunctuation:
      nonQpunct_i.append(i)
    elif text[i] == '?':
      qPunct_i.append(i)

  #if there are no question marks or no punctuation marks other than question marks, then preserve the original form of the text 
  if len(qPunct_i) == 0 or len(nonQpunct_i) == 0:    
    return(text)

  #If there are no non-question punctuation marks after the first occurance of a question mark, 
  #then divide the text into a first half (non question) and a second half (question).  
  #Only use the question half for the question answer algorithm
  elif max(nonQpunct_i) < min(qPunct_i):    
    return(text[max(nonQpunct_i)+2:])

  #If there are question marks at the beginning half of the input, and then no question marks after the first occurence of non-question puctuation,
  #then divide the text into a first half (question) and a second half (non question).
  #Only use the question half for the question answer algorithm
  elif max(qPunct_i) < min(nonQpunct_i):
    return(text[:max(qPunct_i)+1])

  #Can add additional complextity, but for now anything more complex will return the original
  else:
    return(text)
#end seperate_question function

####Scoring Functions####
#function check_username
def check_userName(df):
  '''
  This function checks if the first sentence of the user's input from the input dataframe contains the user's name,
  and returns a text string describing the name combination used.
  Parameters:
      :df: dataframe containing the user's input in df['User']
  Returns:
      :A str representing the user name combination.
  '''
  user_intro = df['User'][0].lower()
  name_first = user_name_first.lower()
  name_last = user_name_last.lower()
  name_full = name_first + ' ' + name_last

  if name_full in user_intro:
    return("full name")
  elif name_first in user_intro:
    return("first name")
  elif name_last in user_intro:
    return("last name")
  else:
    return('none')
#end function check_username

#function check_userJobFunction
def check_userJobFunction(df):
  '''
  This function checks if the first sentence of the user's input from the input dataframe contains the user's job function,
  and returns 1 if true or 0 if false. 
  Parameters:
      :df: dataframe containing the user's input in df['User']
  Returns:
      :1 if the user's job function is included, and 0 if it isn't
  '''
  user_intro = df['User'][0].lower()
  jobFunction = user_jobFunction.lower()

  if jobFunction in user_intro:
    return 1
  else:
    return 0
#end function check_userJobFunction

#function check_userTitle
def check_userTitle(df):
  '''
  This function checks if the first sentence of the user's input from the input dataframe contains the user's title,
  and returns 1 if true or 0 if false. 
  Parameters:
      :df: dataframe containing the user's input in df['User']
  Returns:
      :1 if the user's title is included, and 0 if it isn't
  '''
  user_intro = df['User'][0].lower()
  
  if user_title.lower() in user_intro:
    return 1
  else:
    return 0
#end function check_userTitle

#Function get_greeting
def get_greeting(df):
  '''
  This function checks if the input text contains a "formal" or "informal" greeting, or no greeting.
  Parameters:
      :text: str
  Returns:
      A string describing the existence and formality of the greeting in the input text.
  '''
  greetings_list_formal = ("hello", "good evening", "good afternoon", "good morning", "nice to meet you", "a pleasure to meet you", "good to see you", "greetings", "nice to meet you", 
                "pleased to meet you", "good to meet you")
  
  greetings_list_informal = ("hiya", "howdy", "how's it going", "hi", "morning", "evening", "yo", "what's up", "hey there", "hey", "sup")
  
  user_intro = df['User'][0].lower()

  if any(greeting in user_intro for greeting in greetings_list_formal):
    return "formal"
  elif any(greeting in user_intro for greeting in greetings_list_informal):
    return "informal"
  else:
    return "none"
#end function get_greeting

#Function check_patient_name
def check_patient_name(df):
  '''
  This function checks for the presence of the patient's name and title of a text and returns one of seven name/title combinations as a text string for user input from df['User'].
  Parameters:
      :df: dataframe
      :df['User']: str
  Returns:
      A list of str respresenting the name/title combination of the inputs.
  '''
  title_fullName = patient_title + ' ' + patient_name_full
  title_firstName = patient_title + ' ' + patient_name_first
  title_lastName = patient_title + ' ' + patient_name_last

  user_inputs = df['User']
  patient_addresses = []

  for i in range(len(user_inputs)):
    
    if title_fullName in user_inputs[i]:
      patient_addresses.appended("full name and title")
    elif title_firstName in user_inputs[i]:
      patient_addresses.append("first name and title")
    elif title_lastName in user_inputs[i]:
      patient_addresses.append("last name and title")
    elif patient_name_full in user_inputs[i]:
      patient_addresses.append("full name and no title")
    elif patient_name_first in user_inputs[i]:
      patient_addresses.append("first name and no title")
    elif patient_name_last in user_inputs[i]:
      patient_addresses.append("last name and no title")
    else:
      patient_addresses.append("none")

  return patient_addresses
#end function check_patient_name

#function get_repeatedWords
def get_repeatedWords(df):
  '''
  This function checks all user statements for repetition of words the patient used within their last two (2) statements for every user statement,
  and stores the repeated words and a word count in two new columns within the input dataframe.  The function removes stopwords and words used by the user 
  in their past 2 statements, so the words are meaningful and come from the patient's vocabulary.
  Parameters:
    :df: dataframe
    :Statement.Physician: column in df containing strings for each physician statement
    :Statement.Patient: column in df containing strings for each patient statement
  Returns:
    None, but the dataframe df is editted to include two new columns
  '''
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize

  counts_list = []
  repeatedWords_list = []

  for i in range(len(df['User'])):
    repeatedWords = []
    question = df['User'][i].lower()
    
    if i < 2:
      patient_words = nltk.word_tokenize(' '.join(df['Patient'][0:i]))
      patient_words = [word.lower() for word in patient_words if word.isalnum()]
      patient_words = [word for word in patient_words if not word in stopwords.words()]
      users_words = ' '.join(df['User'][0:i])

      for word in patient_words:
        if word not in users_words.lower():   
          if word in question:
            repeatedWords.append(word)

      repeatedWords_count = len(set(repeatedWords))
      counts_list.append(repeatedWords_count)
      repeatedWords_list.append(list(set(repeatedWords)))

    else:
      patient_words = nltk.word_tokenize(' '.join(df['Patient'][i-2:i]))
      patient_words = [word.lower() for word in patient_words if word.isalnum()]
      patient_words = [word for word in patient_words if not word in stopwords.words()]
      users_words = ' '.join(df['User'][i-2:i])

      for word in patient_words:
        if word not in users_words.lower():
          if word in question:
            repeatedWords.append(word)

      repeatedWords_count = len(set(repeatedWords))
      counts_list.append(repeatedWords_count)
      repeatedWords_list.append(list(set(repeatedWords)))

  df['RepeatedWord_words'] = repeatedWords_list
  df['RepeatedWord_counts'] = counts_list
#end function get_repeatedWords

# function get_polarity sentiment analysis
#from textblob import Textblob as tb
def get_polarity(text):
    '''
    This function returns the polarity of a text on a scale of [-1, 1], with -1
    representing a negative sentiment and 1 representing a positive sentiment.
    Parameters:
        :text: str
    Returns:
        A float representing the polarity of the text.
    '''
    polarity_list = []
    user_inputs = df['User']
    for i in range(len(user_inputs)):
      polarity = tb(user_inputs[i]).sentiment.polarity
      polarity_list.append(polarity)
    return polarity_list

# function get_subjectivity sentiment analysis
def get_subjectivity(text):
    '''
    This function returns the subjectivity of a text on a scale of [0, 1], with the closer 
    to 1 meaning the text is more opinionated rather than factual.
    Parameters:
        :text: str
    Returns:
        A float representing the subjectivity of the text.
    '''
    subjectivity_list = []
    user_inputs = df['User']
    for i in range(len(user_inputs)):
      subjectivity = tb(user_inputs[i]).sentiment.subjectivity
      subjectivity_list.append(subjectivity)
    return subjectivity_list

#function chat
def chat():
  '''
  This function initializes the chat interface for the user to ask questions of the simulated patient, 
  and keeps track of the conversation components in a dataframe, df.
  Parameters:
      :requests input from the user
  Returns:
      :prints responses to the user's questions until the termination keyword 'exit' in input.
  '''

  text = ''
  user_dialogue = []
  patient_dialogue = []
  paired_qs = []
  similarity_scores = []

  unclear_questions = ["I'm not sure what you mean.", "Can you rephrase that?", "I don't know what that means."]
  unclear_index = 0       

  global add_bodypart_count
  add_bodypart_count = 0             

  global patient_name_full
  patient_name_full = "Tom Bradford"
  global patient_name_first
  patient_name_first = "Tom"
  global patient_name_last
  patient_name_last = "Bradford"
  global patient_title
  patient_title = "Mr."

  global bodyPart
  bodyPart = 'none'

  global user_name_first
  user_name_first = input("Please enter your first name: ")
  global user_name_last
  user_name_last = input("Please enter your last name: ")
  global user_title
  user_title = input("Please enter your title (Dr., Nurse, Mrs., etc): ")
  global user_jobFunction
  user_jobFunction = input("Please enter your job function (Nurse Practitioner, ER Attending, etc.): ")

  print("\nThank you, the patient is ready for you. \n\nPatient Information \nName: Tom Bradford \nAge: 71\n\n(Ask a question, or type 'exit' to quit.)\n")
  # Opens microphone input. The input will stop after a keyboard interrupt.
  with MicrophoneStream(rate, chunk) as stream:
    # Uses try method to allow users to manually close the stream
    try:
        # Starts the server connection and thread sending microphone audio
        response_gen = streamclient.start(stream.generator())

        # Iterates through responses and prints them
        for response in response_gen:
            y = json.loads(response)
            
            text = ""
            if y["type"] == "final":
                for i in y["elements"]:
                    text+=i["value"]
                
            if  text=="":
              continue
            print("My speech to text response:",text)
            if(text=="Exit."):
              break
            user_dialogue.append(text)
            text = seperate_question(text)
            get_bodyPart(text)
            response_values = get_response(text)

            response = response_values['response']
            paired_q = response_values['paired_q']
            similarity = response_values['similarity']

            paired_qs.append(paired_q)
            similarity_scores.append(similarity)
            get_bodyPart(response)

            if similarity < 0.80:
              response = unclear_questions[unclear_index]
              unclear_index += 1
              if unclear_index > 2:
                unclear_index = 0

            patient_dialogue.append(response)
            
            print("Patient:", response)
            
            
    except KeyboardInterrupt:
        # Ends the websocket connection.
        streamclient.client.send("EOS")
        pass

  
  print("Thank you!")

  global df
  df = pd.DataFrame()
  df['User'] = user_dialogue
  df['Patient'] = patient_dialogue
  df['Paired_Question'] = paired_qs 
  df['Similarity_Score'] = similarity_scores
  df['Polarity'] = get_polarity(df)
  df['Subjectivity'] = get_subjectivity(df)
  get_repeatedWords(df)
  df['Patient_Named'] = check_patient_name(df)

  global intro_scores
  intro_scores = {}
  intro_scores['userName_Provided'] = check_userName(df)
  intro_scores['userJobFunction_Provided'] = check_userJobFunction(df)
  intro_scores['userTitle_Provided'] = check_userTitle(df)
  intro_scores['Greeting_Provided'] = get_greeting(df)

  Greeting_Provided = intro_scores['Greeting_Provided']
  userJobFunction_Provided = intro_scores['userJobFunction_Provided']
  userName_Provided = intro_scores['userName_Provided']
  userTitle_Provided = intro_scores['userTitle_Provided']

  Intro_scores_list = [Greeting_Provided, userJobFunction_Provided, userName_Provided, userTitle_Provided]
  
  if len(df) > 3:
    while len(Intro_scores_list) < len(df):
      Intro_scores_list.append(0)
    df['Intro_scores'] = Intro_scores_list
  
  print(df)

  #df.to_csv('saveAsUserLastNameAndDate.csv') 
  #files.download('saveAsUserLastNameAndDate.csv')

chat()