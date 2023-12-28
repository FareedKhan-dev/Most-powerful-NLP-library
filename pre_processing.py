import ast
from google.generativeai import embedding

# function to perform tokenization on input text
def tokenize_text(input_text, model):
    # question to be asked
    question = f'''Given the input sentence, perform tokenization on it
    {input_text}
    output must be a list of tokens
    '''

    # generate response
    response = model.generate_content(question)

    # convert string of list to list using ast
    list_str = response.text.replace("True", "True,")
    list_str = list_str.replace("False", "False,")
    list_str = list_str.replace("None", "None,")
    list_str = ast.literal_eval(list_str)
    
    return list_str


# lemmatization function
def lemmatize_text(input_text, model):
    # question to be asked
    question = f'''Given the input sentence, perform Lemmatization on it
    {input_text}
    output must be the lemmatized sentence
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()

# stemming function
def stem_text(input_text, model):
    # question to be asked
    question = f'''Given the input sentence, perform Stemming on it
    {input_text}
    output must be the stemmed sentence
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# function for pattern matching
def extract_patterns(input_text, patterns, model):
    # question to be asked
    question = f'''Given the input sentence, extract {patterns} on it
    {input_text}
    output just contains the extracted pattern and does not contain the pattern name
    '''

    # generate response
    response = model.generate_content(question)
    
    # split the output by newline character
    output_list = response.text.split('\n')

    # remove empty strings from the list
    output_list = [item for item in output_list if item]

    return output_list


# function for cleaning text
def clean_text(input_text, model):
    # question to be asked
    question = f'''Given the input sentence, clean the text but don't change words
    {input_text}
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# function for removing html tags
def remove_html_tags(input_text, tags_to_remove, model):
    # question to be asked
    question = f'''Given the input sentence, remove HTML tags: {tags_to_remove}
    {input_text}
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# function for text replacement
def replace_text(input_text, replacement_rules, model):
    # question to be asked
    question = f'''Given the input sentence, replace words: {replacement_rules}
    {input_text}
    '''

    # generate response
    response = model.generate_content(question)

    return response.text.strip()


# Function for generating embeddings
def extract_embeddings(input_texts):
    
    # embed content
    response = embedding.embed_content(content=input_texts, model="models/embedding-001")
    
    # return response
    return response