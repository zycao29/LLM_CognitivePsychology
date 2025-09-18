####################################### Set up Packages #######################################
import openai # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
###############################################################################################
########################################## Prompt #############################################
###############################################################################################






######################################## Online Models ########################################
############################################# GPT #############################################

#sleep
# prompt_gpt = """

# I will give you a list of words. Please read them carefully.

# Here are the words: bed, rest, tired, awake, dream, blanket, doze, slumber, snore, nap, peace, yawn, drowsy.

# First, Please write a short paragraph based on these words.

# Now, without looking back at the words list, please list the words you remember from the ones I just gave you. 

# """

# prompt_gpt_2 = """

# I will give you a list of words. Please read them carefully.

# Here are the words: bed, rest, tired, awake, dream, blanket, doze, slumber, snore, nap, peace, yawn, drowsy.

# Now, without looking back at the words list, please list the words you remember from the ones I just gave you. 

# """

try_prompt = """
Goal: Memorizing the training list I gave you (composed of several words), and generate a short paragraph related to the words in the training list. After this, I will give you a new test list (also composed of several words), and please extract all the words you remembered from the training list that are in the test list.

Input: The training list and test list. Traning list is [bed, rest, tired, awake, dream, blanket, snore, nap, yawn, drowsy],
Test list is words = [snore, tired, colorful, dream, drowsy, awake, beautiful, nap, blanket, strong, elegant, quiet, fast, mysterious, bright, rest, bed, delicious, yawn, ancient].

Output: The results of the analysis need to be in CSV (comma-separated values). 
Present the results inside a code box.  
The code box should be labeled answers_csv 
The first line is the paragraph you generated, and each subsequent line should list, in order, the words from the test list that you remembered from the training list.

Analysis: 
Here is a template describing how to answer the questions list
- 1: Yes
- 2: No 
"""

def run_gpt():
    print("\nRunning GPT model...\n")
    openai.api_key = ''
    model = "gpt-4-0125-preview"
    input_text = "\n".join(try_prompt)
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "user: " + input_text}
        ]
    )
    reply = response['choices'][0]['message']['content']
    #print("GPT 的回复:")
    print(reply)
















###############################################################################################
############################################ QWen2 ############################################
try_prompt_qwen2 = """

"""


def run_qwen2():
    print("\nRunning Qwen2 model...\n")
    model_name = "Qwen/Qwen2-7B-Instruct"
    device = "cuda" 

    model = AutoModelForCausalLM.from_pretrained(
     model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = try_prompt
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
     tokenize=False,
     add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
     **model_inputs,
     max_new_tokens=2056
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    #response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = tokenizer.batch_decode(generated_ids, min_length=2056)[0]
    print(response)






###############################################################################################
def main():
    parser = argparse.ArgumentParser(description='Select a model to run.')
    parser.add_argument('model', choices=['gpt', 
                                          'qwen2'], help='Model to run')

    args = parser.parse_args()

    if args.model == 'gpt':
        run_gpt()
    elif args.model == 'qwen2':
        run_qwen2()


if __name__ == "__main__":
    main()
