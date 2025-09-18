####################################### Set up Packages #######################################
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
###############################################################################################
########################################## Prompt #############################################
###############################################################################################


###############################################################################################
######################################## Online Models ########################################
############################################# GPT #############################################
# Study    : 32
# Questions: 64
try_prompt_32 = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
"The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", "The detective is in the church.", 
"The maid is in the library.", "The maid is in the studio.", "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", "The tourist is in the nightclub.", "The tourist is in the theater."
]. 
    

Question list is [
"Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", "Is the chef in the attic?", "Is the gardener in the hallway?", "Is the queen in the park?", "Is the coach in the bank?", "Is the hippie in the hotel?",    "Is the scientist in the prison?", "Is the cowboy in the barn?", "Is the inventor in the kitchen?", "Is the sheriff in the school?", "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the soldier in the stadium?", "Is the detective in the church?", "Is the actor in the prison?", "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
"Is the maid in the library?", "Is the spy in the clinic?", "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", "Is the tourist in the factory?"
].


Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

# Study    : 48
# Questions: 96
try_prompt_48 = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 96 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

 
Input: The study list and question list. Study list is ["The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", 
    "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", 
    "The firefighter is in the market.", "The journalist is in the bookstore.", "The lawer is in the aquarium.",
    "The librarian is in the pharmacy.", "The nurse is in the restaurant.", "The photographer is in the station.",
    "The plumber is in the zoo.","The writer is in the bakery.",
    "The accountant is in the beach.","The accountant is in the cinema.","The artist is in the beach.", 
    "The artist is in the cinema.", "The archtist is in the hospital.", "The archtist is in the mall.",
    "The dentist is in the hospital.", "The dentist is in the mall.",
    "The maid is in the library.", "The maid is in the studio.", 
    "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
    "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
    "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
    "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
    "The tourist is in the nightclub.", "The tourist is in the theater."]. 
    
    Question list is ["Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
    "Is the gardener in the hallway?", "Is the coach in the bank?", "Is the inventor in the kitchen?", 
    "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
    "Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
    "Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
    "Is the chef in the attic?", "Is the firefighter in the market?", "Is the journalist in the bookstore?",
    "Is the lawer in the aquarium?","Is the librarian in the pharmacy?","Is the nurse in the restaurant?",
    "Is the photographer in the station?", "Is the plumber in the zoo?","Is the writer in the bakery?",
    "Is the maid in the library?", "Is the spy in the clinic?", 
    "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
    "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
    "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
    "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
    "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the accountant in the beach?",
    "Is the archtist in the hospital?","Is the accountant in the cinema?", "Is the archtist in the mall?",
    "Is the artist in the cinema?","Is the dentist in the hospital?","Is the artist in the beach?",
    "Is the dentist in the mall?",
    "Is the actor in the prison?", 
    "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
    "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
    "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
    "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
    "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
    "Is the firefighter in the bakery?", "Is the journalist in the market?",
    "Is the lawer in the bookstore?","Is the librarian in the aquarium?","Is the nurse in the pharmacy?",
    "Is the photographer in the restaurant?", "Is the plumber in the station?","Is the writer in the zoo?",
    "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
    "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
    "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
    "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
    "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
    "Is the tourist in the factory?",  "Is the accountant in the hospital?",
    "Is the archtist in the beach?","Is the accountant in the mall?", "Is the archtist in the cinema?",
    "Is the artist in the hospital?","Is the dentist in the beach?","Is the artist in the mall?",
    "Is the dentist in the cinema?"].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

###################################### mix list 37 (5 more times) #####################################
########################################## position beginning #########################################
try_prompt_gpt_mix37_begin = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
"The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", "The detective is in the church."
"The maid is in the library.", "The maid is in the studio.", "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", "The tourist is in the nightclub.", "The tourist is in the theater."
].
    

Question list is [
"Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", "Is the chef in the attic?", "Is the gardener in the hallway?", "Is the queen in the park?", "Is the coach in the bank?", "Is the hippie in the hotel?",    "Is the scientist in the prison?", "Is the cowboy in the barn?", "Is the inventor in the kitchen?", "Is the sheriff in the school?", "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the soldier in the stadium?", "Is the detective in the church?", "Is the actor in the prison?", "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?",
"Is the maid in the library?", "Is the spy in the clinic?", "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", "Is the tourist in the factory?"
].


Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

###################################### mix list 37 (5 more times) #####################################
########################################### position middle ###########################################
try_prompt_gpt_mix37_fan1_middle = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
    "The farmer is in the garage.", "The pirate is in the office.", 
    "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.",
    "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.",
    "The maid is in the library.", "The maid is in the studio.", 
    "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
    "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
    "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
    "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
    "The tourist is in the nightclub.", "The tourist is in the theater."]. 
    
    Question list is [
    "Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
    "Is the gardener in the hallway?", "Is the coach in the bank?", "Is the inventor in the kitchen?", 
    "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
    "Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
    "Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
    "Is the chef in the attic?", "Is the maid in the library?", "Is the spy in the clinic?", 
    "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
    "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
    "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
    "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
    "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the actor in the prison?", 
    "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
    "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
    "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
    "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
    "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
    "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
    "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
    "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
    "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
    "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
    "Is the tourist in the factory?"].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""


###################################### mix list 37 (5 more times) #####################################
############################################# position end ############################################
try_prompt_gpt_mix37_fan1_end = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
    "The farmer is in the garage.", "The pirate is in the office.", 
    "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", 
    "The maid is in the library.", "The maid is in the studio.", 
    "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
    "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
    "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
    "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
    "The tourist is in the nightclub.", "The tourist is in the theater.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.",
    "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.",]. 
    
    Question list is [
    "Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
    "Is the gardener in the hallway?", "Is the coach in the bank?", "Is the inventor in the kitchen?", 
    "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
    "Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
    "Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
    "Is the chef in the attic?", "Is the maid in the library?", "Is the spy in the clinic?", 
    "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
    "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
    "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
    "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
    "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the actor in the prison?", 
    "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
    "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
    "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
    "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
    "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
    "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
    "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
    "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
    "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
    "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
    "Is the tourist in the factory?"].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

###################################### mix list 37 (5 more times) #####################################
########################################## position 25% #########################################
try_prompt_gpt_mix37_25percent_5times = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
"The farmer is in the garage.", "The pirate is in the office.", "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", "The detective is in the church."
"The maid is in the library.", "The maid is in the studio.", "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", "The tourist is in the nightclub.", "The tourist is in the theater."
].
    

Question list is [
"Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", "Is the chef in the attic?", "Is the gardener in the hallway?", "Is the queen in the park?", "Is the coach in the bank?", "Is the hippie in the hotel?",    "Is the scientist in the prison?", "Is the cowboy in the barn?", "Is the inventor in the kitchen?", "Is the sheriff in the school?", "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the soldier in the stadium?", "Is the detective in the church?", "Is the actor in the prison?", "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?",
"Is the maid in the library?", "Is the spy in the clinic?", "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", "Is the tourist in the factory?"
].


Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""


###################################### mix list 37 (5 more times) #####################################
########################################## position 50% #########################################
try_prompt_gpt_mix37_50percent_5times = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
"The farmer is in the garage.", "The pirate is in the office.", "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", "The detective is in the church.","The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.",
"The maid is in the library.", "The maid is in the studio.", "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", "The tourist is in the nightclub.", "The tourist is in the theater."
].
    

Question list is [
"Is the maid in the library?", "Is the spy in the clinic?", "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", "Is the tourist in the factory?",
"Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", "Is the chef in the attic?", "Is the gardener in the hallway?", "Is the queen in the park?", "Is the coach in the bank?", "Is the hippie in the hotel?",    "Is the scientist in the prison?", "Is the cowboy in the barn?", "Is the inventor in the kitchen?", "Is the sheriff in the school?", "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the soldier in the stadium?", "Is the detective in the church?", "Is the actor in the prison?", "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?"
].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""



###################################### mix list 37 (5 more times) #####################################
########################################## position 75% #########################################
try_prompt_gpt_mix37_75percent_5times = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
"The farmer is in the garage.", "The pirate is in the office.", "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", "The detective is in the church.",
"The maid is in the library.", "The maid is in the studio.", "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.", "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", "The tourist is in the nightclub.", "The tourist is in the theater."
].
    

Question list is [
"Is the maid in the library?", "Is the spy in the clinic?", "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", "Is the tourist in the factory?",
"Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", "Is the chef in the attic?", "Is the gardener in the hallway?", "Is the queen in the park?", "Is the coach in the bank?", "Is the hippie in the hotel?",    "Is the scientist in the prison?", "Is the cowboy in the barn?", "Is the inventor in the kitchen?", "Is the sheriff in the school?", "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the soldier in the stadium?", "Is the detective in the church?", "Is the actor in the prison?", "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?"
].


Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""


###################################### mix list 42 (10 more times) #####################################
########################################## position beginning #########################################
try_prompt_gpt_mix42_begin = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
    "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.",
    "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.",
    "The actor is in the airport.", "The actor is in the airport.",
    "The actor is in the airport.", "The actor is in the airport.", "The actor is in the airport.",
    "The farmer is in the garage.", "The pirate is in the office.", 
    "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", 
    "The maid is in the library.", "The maid is in the studio.", 
    "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
    "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
    "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
    "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
    "The tourist is in the nightclub.", "The tourist is in the theater."]. 
    
    Question list is [
    "Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
    "Is the gardener in the hallway?", "Is the coach in the bank?", "Is the inventor in the kitchen?", 
    "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
    "Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
    "Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
    "Is the chef in the attic?", "Is the maid in the library?", "Is the spy in the clinic?", 
    "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
    "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
    "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
    "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
    "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the actor in the prison?", 
    "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
    "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
    "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
    "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
    "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
    "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
    "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
    "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
    "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
    "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
    "Is the tourist in the factory?"].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""


###################################### mix list 37 (5 more times) #####################################
########################################## position beginning #########################################
try_prompt_gpt_mix37_fan2_begin = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
    "The maid is in the library.","The maid is in the library.","The maid is in the library.",
    "The maid is in the library.","The maid is in the library.","The maid is in the library.",
    "The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", 
    "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", 
    "The maid is in the studio.", "The spy is in the clinic.", "The spy is in the museum.", 
    "The doctor is in the temple.", "The doctor is in the factory.", "The musician is in the nightclub.", 
    "The musician is in the theater.", "The teacher is in the library.", "The teacher is in the studio.", 
    "The engineer is in the clinic.", "The engineer is in the museum.", "The pilot is in the temple.",
    "The pilot is in the factory.", "The tourist is in the nightclub.", "The tourist is in the theater."
    ]. 
    
    Question list is [
    "Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", "Is the chef in the attic?", "Is the gardener in the hallway?", "Is the queen in the park?", "Is the coach in the bank?", "Is the hippie in the hotel?",    "Is the scientist in the prison?", "Is the cowboy in the barn?", "Is the inventor in the kitchen?", "Is the sheriff in the school?", "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the soldier in the stadium?", "Is the detective in the church?", "Is the actor in the prison?", "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?",
    "Is the maid in the library?", "Is the spy in the clinic?", "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", "Is the tourist in the factory?"
    ].



Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

###################################### mix list 42 (10 more times) #####################################
########################################## position beginning #########################################
try_prompt_gpt_mix42_fan2_begin = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
    "The maid is in the library.","The maid is in the library.","The maid is in the library.",
    "The maid is in the library.","The maid is in the library.","The maid is in the library.",
    "The maid is in the library.","The maid is in the library.",
    "The maid is in the library.","The maid is in the library.","The maid is in the library.",
    "The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", 
    "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", 
    "The maid is in the studio.", 
    "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
    "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
    "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
    "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
    "The tourist is in the nightclub.", "The tourist is in the theater."]. 
    
    Question list is [
    "Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
    "Is the gardener in the hallway?", "Is the coach in the bank?", "Is the inventor in the kitchen?", 
    "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
    "Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
    "Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
    "Is the chef in the attic?", "Is the maid in the library?", "Is the spy in the clinic?", 
    "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
    "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
    "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
    "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
    "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the actor in the prison?", 
    "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
    "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
    "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
    "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
    "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
    "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
    "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
    "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
    "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
    "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
    "Is the tourist in the factory?"].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""


############################################ pure list 32 ###########################################

try_prompt_DRM_5_fan1 = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
    "The actor is in the hangar.", "The farmer is in the carport.", "The pirate is in the cubicle.", 
    "The chef is in the loft.", "The gardener is in the foyer.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", 
    "The maid is in the library.", "The maid is in the studio.", 
    "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
    "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
    "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
    "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
    "The tourist is in the nightclub.", "The tourist is in the theater."]. 
    
    Question list is [
    "Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
    "Is the chef in the attic?","Is the gardener in the hallway?", "Is the coach in the bank?", 
    "Is the inventor in the kitchen?", 
    "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
    "Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
    "Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
    "Is the maid in the library?", "Is the spy in the clinic?", 
    "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
    "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
    "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
    "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
    "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the actor in the prison?", 
    "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
    "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
    "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
    "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
    "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
    "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
    "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
    "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
    "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
    "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
    "Is the tourist in the factory?"].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""


##################################  Replace regardless of semantic #################################
####################################################################################################
try_prompt_32_repalce_regardless_fan1 = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
"The actor is in the wJP4hDl.", "The farmer is in the cK710y.", "The pirate is in the 9avvBC.", "The chef is in the kCzVP.", "The gardener is in the TODJ7Me.", "The queen is in the M1p3.", "The coach is in the Px62.", "The hippie is in the HJzQT.", "The scientist is in the QygP1U.", "The cowboy is in the 5Jpt.", "The inventor is in the vDi6efy.", "The sheriff is in the gMXXK6.", "The dancer is in the plNWGl.", "The judge is in the hID6AcneDb.", "The soldier is in the 6O6bMZ6.", "The detective is in the J4fxHG.", 
"The maid is in the library.", "The maid is in the studio.", "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", "The tourist is in the nightclub.", "The tourist is in the theater."
]. 
    

Question list is [
"Is the actor in the wJP4hDl?", "Is the farmer in the cK710y?", "Is the pirate in the 9avvBC?", "Is the chef in the kCzVP?", "Is the gardener in the TODJ7Me?", "Is the queen in the M1p3?", "Is the coach in the Px62?", "Is the hippie in the HJzQT?",    "Is the scientist in the QygP1U?", "Is the cowboy in the 5Jpt?", "Is the inventor in the vDi6efy?", "Is the sheriff in the gMXXK6?", "Is the dancer in the plNWGl?", "Is the judge in the hID6AcneDb?", "Is the soldier in the 6O6bMZ6?", "Is the detective in the J4fxHG?", "Is the actor in the QygP1U?", "Is the farmer in the kCzVP?", "Is the pirate in the cK710y?", "Is the gardener in the gMXXK6?", "Is the coach in the TODJ7Me?", "Is the inventor in the 6O6bMZ6?", "Is the dancer in the J4fxHG?", "Is the judge in the vDi6efy?", "Is the hippie in the plNWGl?", "Is the scientist in the 5Jpt?", "Is the soldier in the HJzQT?", "Is the detective in the wJP4hDl?", "Is the cowboy in the hID6AcneDb?", "Is the queen in the Px62?", "Is the sheriff in the 9avvBC?", "Is the chef in the M1p3?", 
"Is the maid in the library?", "Is the spy in the clinic?", "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", "Is the tourist in the factory?"
].


Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

try_prompt_32_repalce_regardless_fan2 =  """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
"The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", "The detective is in the church.", 
"The maid is in the su1iu9o.", "The maid is in the a2sdc9.", "The spy is in the 1l1f3g.", "The spy is in the 8e9h1n.", "The doctor is in the z8w2j9.", "The doctor is in the q4r5b3.", "The musician is in the g8h4v2.", "The musician is in the o5q9l7.", "The teacher is in the su1iu9o.", "The teacher is in the a2sdc9.", "The engineer is in the 1l1f3g.", "The engineer is in the 8e9h1n.", "The pilot is in the z8w2j9.", "The pilot is in the q4r5b3.", "The tourist is in the g8h4v2.", "The tourist is in the o5q9l7."
]. 
    

Question list is [
"Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", "Is the chef in the attic?", "Is the gardener in the hallway?", "Is the queen in the park?", "Is the coach in the bank?", "Is the hippie in the hotel?",    "Is the scientist in the prison?", "Is the cowboy in the barn?", "Is the inventor in the kitchen?", "Is the sheriff in the school?", "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the soldier in the stadium?", "Is the detective in the church?", "Is the actor in the prison?", "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
"Is the maid in the su1iu9o?", "Is the spy in the 1l1f3g?", "Is the doctor in the z8w2j9?", "Is the musician in the g8h4v2?", "Is the teacher in the a2sdc9?", "Is the engineer in the 8e9h1n?", "Is the pilot in the q4r5b3?", "Is the tourist in the o5q9l7?", "Is the maid in the a2sdc9?", "Is the spy in the 8e9h1n?", "Is the doctor in the q4r5b3?", "Is the musician in the o5q9l7?", "Is the teacher in the su1iu9o?", "Is the engineer in the 1l1f3g?", "Is the pilot in the z8w2j9?", "Is the tourist in the g8h4v2?", "Is the maid in the g8h4v2?", "Is the spy in the su1iu9o?", "Is the doctor in the 1l1f3g?", "Is the musician in the q4r5b3?", "Is the teacher in the o5q9l7?", "Is the engineer in the a2sdc9?", "Is the pilot in the 8e9h1n?", "Is the tourist in the z8w2j9?", "Is the maid in the o5q9l7?", "Is the spy in the a2sdc9?", "Is the doctor in the 8e9h1n?", "Is the musician in the z8w2j9?", "Is the teacher in the g8h4v2?", "Is the engineer in the su1iu9o?", "Is the pilot in the 1l1f3g?", "Is the tourist in the q4r5b3?"
].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

###################################  shuffle the pure study list ###################################
####################################################################################################
try_prompt_32_shuffle = """
Goal: Memorizing the study list, where all sentences are in the format of "The person is in the location", answer all 64 questions in the questions list (questions are in the form of "Is the person in the location?") based on the study list. If the person-location pair is found in the study list, answer "yes"; otherwise, answer "no."

Input: The study list and question list. Study list is [  
'The pilot is in the factory.','The detective is in the church.', 'The actor is in the airport.', 'The teacher is in the library.', 'The musician is in the nightclub.', 'The pilot is in the factory.','The spy is in the museum.', 'The engineer is in the clinic.', 'The gardener is in the hallway.', 'The doctor is in the temple.', 'The tourist is in the nightclub.', 'The queen is in the park.','The pilot is in the factory.', 'The inventor is in the kitchen.', 'The hippie is in the hotel.', 'The maid is in the studio.', 'The dancer is in the castle.', 'The cowboy is in the barn.', 'The spy is in the clinic.', 'The pilot is in the factory.','The pilot is in the temple.', 'The musician is in the theater.', 'The chef is in the attic.', 'The judge is in the laboratory.', 'The coach is in the bank.', 'The teacher is in the studio.', 'The maid is in the library.', 'The pirate is in the office.', 'The soldier is in the stadium.','The pilot is in the factory.', 'The engineer is in the museum.', 'The scientist is in the prison.', 'The sheriff is in the school.', 'The doctor is in the factory.', 'The pilot is in the factory.','The tourist is in the theater.', 'The farmer is in the garage.'
]. 
    

Question list is [
"Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", "Is the chef in the attic?", "Is the gardener in the hallway?", "Is the queen in the park?", "Is the coach in the bank?", "Is the hippie in the hotel?",    "Is the scientist in the prison?", "Is the cowboy in the barn?", "Is the inventor in the kitchen?", "Is the sheriff in the school?", "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the soldier in the stadium?", "Is the detective in the church?", "Is the actor in the prison?", "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
"Is the maid in the library?", "Is the spy in the clinic?", "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", "Is the tourist in the factory?"
].


Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""



prompt1 = """
Goal: Memorizing the study list, where the study list consists of person-location pairs (formatted as "the person is in the location."). Please answer all 64 questions in the question list. If the person-location pair from the questions appears in the study list, respond with "yes"; otherwise, respond with "no."

Input: The study list and question list. Study list is [  
The actor is in the airport.
The actor is in the attic.
The chef is in the airport.
The chef is in the attic.
The coach is in the bank.
The coach is in the barn.
The cowboy is in the bank.
The cowboy is in the barn.
The dancer is in the castle.
The dancer is in the church.
The detective is in the castle.
The detective is in the church.
The doctor is in the clinic.
The doctor is in the factory.
The engineer is in the clinic.
The engineer is in the factory.
The farmer is in the garage.
The farmer is in the hallway.
The gardener is in the garage.
The gardener is in the hallway.
The hippie is in the hotel.
The hippie is in the kitchen.
The inventor is in the hotel.
The inventor is in the kitchen.
The judge is in the laboratory.
The judge is in the library.
The maid is in the laboratory.
The maid is in the library.
The musician is in the museum.
The musician is in the nightclub.
The pilot is in the museum.
The pilot is in the nightclub.
]
    
Question list is [
Is the actor in the airport?
Is the coach in the bank?
Is the chef in the airport?
Is the cowboy in the barn? 
Is the actor in the attic?
Is the coach in the barn?
Is the cowboy in the bank? 
Is the chef in the attic?
Is the detective in the castle?
Is the doctor in the clinic? 
Is the dancer in the church?
Is the engineer in the clinic?
Is the dancer in the castle?
Is the doctor in the factory?
Is the detective in the church?
Is the engineer in the factory?
Is the farmer in the garage?
Is the hippie in the kitchen? 
Is the inventor in the hotel?
Is the gardener in the hallway?
Is the hippie in the hotel?
Is the farmer in the hallway?
Is the inventor in the kitchen?
Is the gardener in the garage? 
Is the musician in the museum?
Is the judge in the library?
Is the maid in the laboratory?
Is the pilot in the nightclub? 
Is the judge in the laboratory?
Is the musician in the nightclub?
Is the pilot in the museum?
Is the maid in the library?
Is the actor in the garage?
Is the coach in the hotel?
Is the chef in the garage?
Is the cowboy in the kitchen? 
Is the actor in the hallway?
Is the coach in the kitchen?
Is the cowboy in the hotel? 
Is the chef in the hallway?
Is the detective in the laboratory?
Is the doctor in the museum? 
Is the dancer in the library?
Is the engineer in the museum?
Is the dancer in the laboratory?
Is the doctor in the nightclub?
Is the detective in the library?
Is the engineer in the nightclub?
Is the farmer in the airport?
Is the hippie in the barn? 
Is the inventor in the bank?
Is the gardener in the attic?
Is the hippie in the bank?
Is the farmer in the attic?
Is the inventor in the barn?
Is the gardener in the airport?
Is the musician in the clinic?
Is the judge in the church?
Is the maid in the castle?
Is the pilot in the factory? 
Is the judge in the castle?
Is the musician in the factory?
Is the pilot in the clinic?
Is the maid in the church?
].


Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order of each question from the question list and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

prompt2 = """
Goal: Determine if each person-location pair from the question list appears in the study list. 

For each of 64 question, respond with the question number followed by a question itslef and then 'Yes' if the pair is presented in the study list once, and 'No' otherwise.

questions:
Is the 2s3fcin the a5gsd3?,
Is the 1wxain the 9df2c?,
Is the sd2lj in the s58t?,
Is the 2sce8q in the 72sr?,
Is the cni7e9 in the m0dg2a?,
Is the n53wc54zq in the 8nm43x?,
Is the a5po92 in the 0i8c3a?,
Is the s2c2ay70 in the pms53sq?,
Is the 9f3nai in the 9mct42?,
Is the 2rzh62aq in the nsu28fg?,
Is the af01gd in the u7c6w?,
Is the cp29b35f in the 2dt8z6s?,
Is the t56ui in the i3n0fj3mfp?,
Is the rf5h in the 9sj23fr?,
Is the b629e0ki in the u827sc?,
Is the rg0pq in the 51gtqcq8h?,
Is the 91lsnc in the 1ms2ce?,
Is the 29uzd in the ou8y?,
Is the o9cena75g in the mi29uh?,
Is the e134cma in the 8fg35g?,
Is the 8idm23c in the zqp23yn?,
Is the 3n8 in the ap092c?,
Is the cuqp125 in the 8thj14?,
Is the n83hyaz in the mwe30ci?,
Is the 83dna3k9cy in the 34ftv?,
Is the 4ftv1s in the qmfl0k?,
Is the 37ycuaq in the ml210sxq?,
Is the 5aydf in the 48cisu26ct?,
Is the a6d4f in the 4ncvb28?,
Is the q6egv14 in the x47j?,
Is the q6w55be4f1 in the 6gl?,
Is the 294grh in the mi20dac9?,
Is the 2s3fcin the mi20dac9?,
Is the 1wxain the 6gl?,
Is the sd2lj in the x47j?,
Is the 2sce8q in the 4ncvb28?,
Is the cni7e9 in the 48cisu26ct?,
Is the n53wc54zq in the ml210sxq?,
Is the a5po92 in the qmfl0k?,
Is the s2c2ay70 in the 34ftv?,
Is the 9f3nai in the mwe30ci?,
Is the 2rzh62aq in the 8thj14?,
Is the af01gd in the ap092c?,
Is the cp29b35f in the zqp23yn?,
Is the t56ui in the 8fg35g?,
Is the rf5h in the mi29uh?,
Is the b629e0ki in the ou8y?,
Is the rg0pq in the 1ms2ce?,
Is the 91lsnc in the 51gtqcq8h?,
Is the 29uzd in the u827sc?,
Is the o9cena75g in the 9sj23fr?,
Is the e134cma in the i3n0fj3mfp?,
Is the 8idm23c in the 2dt8z6s?,
Is the 3n8 in the u7c6w?,
Is the cuqp125 in the nsu28fg?,
Is the n83hyaz in the 9mct42?,
Is the 83dna3k9cy in the pms53sq?,
Is the 4ftv1s in the 0i8c3a?,
Is the 37ycuaq in the 8nm43x?,
Is the 5aydf in the m0dg2a?,
Is the a6d4f in the 72sr?,
Is the q6egv14 in the s58t?,
Is the q6w55be4f1 in the 9df2c?,
Is the 294grh in the a5gsd3?



study_list:
The 2s3fc is in the a5gsd3,
The 1wxa is in the 9df2c,
The sd2lj is in the s58t,
The 2sce8q is in the 72sr,
The cni7e9 is in the m0dg2a,
The n53wc54zq is in the 8nm43x,
The a5po92 is in the 0i8c3a,
The s2c2ay70 is in the pms53sq,
The 9f3nai is in the 9mct42,
The 2rzh62aq is in the nsu28fg,
The af01gd is in the u7c6w,
The cp29b35f is in the 2dt8z6s,
The t56ui is in the i3n0fj3mfp,
The rf5h is in the 9sj23fr,
The b629e0ki is in the u827sc,
The rg0pq is in the 51gtqcq8h,
The 91lsnc is in the 1ms2ce,
The 29uzd is in the ou8y,
The o9cena75g is in the mi29uh,
The e134cma is in the 8fg35g,
The 8idm23c is in the zqp23yn,
The 3n8 is in the ap092c,
The cuqp125 is in the 8thj14,
The n83hyaz is in the mwe30ci,
The 83dna3k9cy is in the 34ftv,
The 4ftv1s is in the qmfl0k,
The 37ycuaq is in the ml210sxq,
The 5aydf is in the 48cisu26ct,
The a6d4f is in the 4ncvb28,
The q6egv14 is in the x47j,
The q6w55be4f1 is in the 6gl,
The 294grh is in the mi20dac9
"""



queue1 = """
Goal: Memorizing the study list, where the study list consists of person-location pairs (formatted as "the person is in the location."). Please recall and list all the person-location pairs you have memorized in the study list. 

Input: The study list and question list. Study list is [  
The actor is in the airport,
The actor is in the airport,
The actor is in the airport,
The actor is in the airport,
The actor is in the airport,
The chef is in the attic,
The coach is in the bank,
The cowboy is in the barn,
The dancer is in the castle,
The detective is in the church,
The doctor is in the clinic,
The engineer is in the factory,
The farmer is in the garage,
The gardener is in the hallway,
The hippie is in the hotel,
The inventor is in the kitchen,
The judge is in the laboratory,
The maid is in the library,
The musician is in the museum,
The pilot is in the nightclub,
The pirate is in the office,
The queen is in the park,
The scientist is in the prison,
The sheriff is in the school,
The soldier is in the stadium,
The spy is in the studio,
The teacher is in the temple,
The tourist is in the theater,
The accountant is in the beach,
The artist is in the cinema,
The dentist is in the hospital,
The nurse is in the restaurant,
The writer is in the station,
The barista is in the mall,
The firefighter is in the zoo,
The lawyer is in the pharmacy
]. 
    


Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
Each line of the output should contain the order and the corresponding answer.

Analysis: 
Here is a template describing how to answer the questions list
- 1: actor, airport
- 2: 
"""







def run_gpt():
    print("\nRunning GPT model...\n")
    openai.api_key = ''
    model = "gpt-4-0125-preview"
    ##"gpt-4-0125-preview"
    input_text = "\n".join(queue1)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "user: " + input_text}
            
        ],
        #temperature=0.6,
        #top_p=0.9,
    )
    reply = response['choices'][0]['message']['content']
    #print("GPT 的回复:")
    print(reply)









###############################################################################################
###############################################################################################
###############################################################################################
############################################ QWen2 ############################################
###############################################################################################
###############################################################################################
###############################################################################################

try_prompt_qwen2 = """
Study List:
    "The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", 
    "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", "The maid is in the library.", "The maid is in the studio.", 
    "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
    "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
    "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
    "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
    "The tourist is in the nightclub.", "The tourist is in the theater."
Questions:
    "Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
    "Is the gardener in the hallway?", "Is the coach in the bank?", "Is the inventor in the kitchen?", 
    "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
    "Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
    "Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
    "Is the chef in the attic?", "Is the maid in the library?", "Is the spy in the clinic?", 
    "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
    "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
    "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
    "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
    "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the actor in the prison?", 
    "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
    "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
    "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
    "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
    "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
    "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
    "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
    "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
    "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
    "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
    "Is the tourist in the factory?".
"""





def run_qwen2():
    print("\nRunning Qwen2 model...\n")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda" 

    model = AutoModelForCausalLM.from_pretrained(
     model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": try_prompt_qwen2}
    ]
    text = tokenizer.apply_chat_template(
        messages,
     tokenize=False,
     add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
     **model_inputs,
     max_new_tokens=3000
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    #response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = tokenizer.batch_decode(generated_ids, min_length=2056)[0]
    print(response)




###############################################################################################
###############################################################################################
############################################         ##########################################
############################################ Offline ##########################################
############################################         ##########################################
###############################################################################################
###############################################################################################


###############################################################################################
############################################ Llama2 ###########################################
try_prompt_llama2 = """
Study and remember the following list of sentences:
["The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", 
"The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
"The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
"The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
"The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
"The detective is in the church.", 
"The maid is in the library.", "The maid is in the studio.", 
"The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
"The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
"The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
"The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
"The tourist is in the nightclub.", "The tourist is in the theater."]. 
Then, answer the following questions:
["Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
"Is the gardener in the hallway?", "Is the coach in the bank?", "Is the inventor in the kitchen?", 
"Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
"Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
"Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
"Is the chef in the attic?", "Is the maid in the library?", "Is the spy in the clinic?", 
"Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
"Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
"Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
"Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
"Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the actor in the prison?", 
"Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
"Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
"Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
"Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
"Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
"Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
"Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
"Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
"Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
"Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
"Is the tourist in the factory?"]
Respond 'yes' if the person-location pair appears in the study list, and 'no' otherwise."
"""


def run_llama2():
    print("\nRunning Llama2 model...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('./llama2_tokenizer')
    #model = AutoModelForCausalLM.from_pretrained('./llama2_model')
    model = AutoModelForCausalLM.from_pretrained('./llama2_model',device_map="auto")
    #model = model.to(device)
    prompt = try_prompt_llama2
    #prompt = 'how are you'
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)





###############################################################################################
############################################ mistral ##########################################

try_prompt_mistral = """
"Remember the given list of sentences in the format 'the person is in the location': 
["The actor is in the airport.", "The farmer is in the garage.", "The pirate is in the office.", 
    "The chef is in the attic.", "The gardener is in the hallway.", "The queen is in the park.", 
    "The coach is in the bank.", "The hippie is in the hotel.", "The scientist is in the prison.", 
    "The cowboy is in the barn.", "The inventor is in the kitchen.", "The sheriff is in the school.", 
    "The dancer is in the castle.", "The judge is in the laboratory.", "The soldier is in the stadium.", 
    "The detective is in the church.", 
    "The maid is in the library.", "The maid is in the studio.", 
    "The spy is in the clinic.", "The spy is in the museum.", "The doctor is in the temple.", 
    "The doctor is in the factory.", "The musician is in the nightclub.", "The musician is in the theater.", 
    "The teacher is in the library.", "The teacher is in the studio.", "The engineer is in the clinic.", 
    "The engineer is in the museum.", "The pilot is in the temple.", "The pilot is in the factory.", 
    "The tourist is in the nightclub.", "The tourist is in the theater."]. 
Answer yes or no to subsequent questions based on whether the person-location pair appears in the study list. 
Example questions list include: 
["Is the actor in the airport?", "Is the farmer in the garage?", "Is the pirate in the office?", 
    "Is the gardener in the hallway?", "Is the coach in the bank?", "Is the inventor in the kitchen?", 
    "Is the dancer in the castle?", "Is the judge in the laboratory?", "Is the hippie in the hotel?", 
    "Is the scientist in the prison?", "Is the soldier in the stadium?", "Is the detective in the church?", 
    "Is the cowboy in the barn?", "Is the queen in the park?", "Is the sheriff in the school?", 
    "Is the chef in the attic?", "Is the maid in the library?", "Is the spy in the clinic?", 
    "Is the doctor in the temple?", "Is the musician in the nightclub?", "Is the teacher in the studio?", 
    "Is the engineer in the museum?", "Is the pilot in the factory?", "Is the tourist in the theater?", 
    "Is the maid in the studio?", "Is the spy in the museum?", "Is the doctor in the factory?", 
    "Is the musician in the theater?", "Is the teacher in the library?", "Is the engineer in the clinic?", 
    "Is the pilot in the temple?", "Is the tourist in the nightclub?", "Is the actor in the prison?", 
    "Is the farmer in the attic?", "Is the pirate in the garage?", "Is the gardener in the school?", 
    "Is the coach in the hallway?", "Is the inventor in the stadium?", "Is the dancer in the church?", 
    "Is the judge in the kitchen?", "Is the hippie in the castle?", "Is the scientist in the barn?", 
    "Is the soldier in the hotel?", "Is the detective in the airport?", "Is the cowboy in the laboratory?", 
    "Is the queen in the bank?", "Is the sheriff in the office?", "Is the chef in the park?", 
    "Is the maid in the nightclub?", "Is the spy in the library?", "Is the doctor in the clinic?", 
    "Is the musician in the factory?", "Is the teacher in the theater?", "Is the engineer in the studio?", 
    "Is the pilot in the museum?", "Is the tourist in the temple?", "Is the maid in the theater?", 
    "Is the spy in the studio?", "Is the doctor in the museum?", "Is the musician in the temple?", 
    "Is the teacher in the nightclub?", "Is the engineer in the library?", "Is the pilot in the clinic?", 
    "Is the tourist in the factory?"]
"""

def run_mistral():
    print("\nRunning Mistral model...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('./mistral_tokenizer')
    model = AutoModelForCausalLM.from_pretrained('./mistral_model')
    model = model.to(device)
    prompt = try_prompt_mistral
    
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=2056)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)





###############################################################################################
def main():
    parser = argparse.ArgumentParser(description='Select a model to run.')
    parser.add_argument('model', choices=['gpt', 
                                          'qwen2',
                                          'llama2',
                                          'mistral'], help='Model to run')

    args = parser.parse_args()

    if args.model == 'gpt':
        run_gpt()
    if args.model == 'qwen2':
        run_qwen2()
    if args.model == 'llama2':
        run_llama2()
    elif args.model == 'mistral':
        run_mistral()

if __name__ == "__main__":
    main()
