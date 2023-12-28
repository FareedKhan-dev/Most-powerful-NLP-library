![Header Image)](https://i.ibb.co/thJLdDt/Group-76.png)


Consider one of the most stressful scenarios that most coders face, dealing with large text data that requires cleaning. When using regex, you need to define different sets of patterns to remove text, and even then, you may not be sure if there is any new garbage data that needs removal. Tasks like these can be stressful for developers because of the time and effort they have to invest, and there’s still uncertainty about whether the new data requires the same procedural coding or not.

The recent trends of Large Language Models (LLMs), whether open source or closed source, have given us a new dimension of how text data can be handled. Since LLMs can analyze text data more quickly than us and can intelligently understand data to a considerable extent, similar to the way we understand it, why don’t we perform our NLP tasks using LLMs? This could automate the process and make the coder’s life less stressful.

## The Core Concept Driving My Library

I tried out several open-source LLMs like [Mistral 8x7b](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) or [LLAMA-2–70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf), but they never met my expectations. Although good for question answering and text generation, they fall short when it comes to NLP tasks. On the other hand, [ChatGPT](https://openai.com/blog/chatgpt) exceeded my expectations, but it requires a paid API access to perform NLP tasks on our custom dataset. [Gemini](https://deepmind.google/technologies/gemini/), as capable as GPT-4, provides a free API with limited access. I tested it with the help of prompt engineering and found that it can solve almost any NLP task you want to tackle.

Here is a simple visual illustration of how I have used Gemini multi-model to perform NLP tasks on my dataset:

![Visual illustration of how my library works ( [Created using FIGMA](https://www.figma.com/) )](https://cdn-images-1.medium.com/max/3020/1*BmW2FzclMY-pghvFuMTNDg.png)

## list of features avaliable in my library

for task related to preprocessing

| Preprocessing Function          | Description                                                                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| tokenize_text           | Given an input sentence, perform tokenization and return a list of tokens.                               |
| lemmatize_text          | Given an input sentence, perform Lemmatization and return the lemmatized sentence.                         |
| stem_text               | Given an input sentence, perform Stemming and return the stemmed sentence.                                 |
| extract_patterns        | Given an input sentence, extract specified patterns and return a list of extracted patterns.               |
| clean_text              | Given an input sentence, clean the text without changing words.                                           |
| remove_html_tags        | Given an input sentence, remove specified HTML tags.                                                      |
| replace_text            | Given an input sentence, replace specified words using replacement rules.                                  |
| extract_embeddings      | Given a list of input texts, generate embeddings using a specified model.                                  |

for task not related to preprocessing

| Core NLP Function          | Description                                                                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------|
| analyze_sentiment       | Perform sentiment analysis on the input text and return the sentiment category.                           |
| classify_topic          | Perform Topic Classification on the input text and return the topic category.                               |
| detect_spam             | Perform spam detection on the input text and return the spam category.                                      |
| detect_ner              | Perform NER detection on the input text and return entities in the format word: entity.                    |
| detect_pos              | Perform POS detection on the input text and return words with their respective POS tags.                   |
| translate_text          | Translate the input text from a source language to a target language.                                       |
| summarize_text          | Summarize the input text to the specified length.                                                          |
| answer_question         | Answer a given question based on the input text.                                                           |
| generate_text           | Generate a text of specified length based on a prompt.                                                     |
| perform_srl             | Perform Semantic Role Labeling (SRL) detection on the input text and return predicates and roles.          |
| recognize_intent        | Perform Intent Recognition detection on the input text and return the detected intent.                    |
| paraphrasing_detection  | Determine if two sentences are paraphrases and return 'yes' or 'no' with an optional explanation.         |


**Understanding File Structure**

You can skip this step as it is for later use only if you want to understand the library and how it works. Here is the file structure of this library.

    main_directory/
    |-- for_beginner/
    |   |-- preprocessing.ipynb
    |   |-- core_nlp.ipynb
    |-- pre_processing.py
    |-- core_nlp.py
    |-- code_file.ipyb  # Containing example of each functions

A “**for_beginner**” folder containing two Jupyter notebook files with code blocks for each NLP task that will make it easier for you to understand how this library is working. While both Python files are going to be used to call as modules and use them for your requested task.

pre_processing.py contains functions that are used for preprocessing our text, such as clean_text, remove_html_tags, etc., while core_nlp contains functions that are useful for handling text data and performing different tasks, such as summarize_text, translate_text, etc.

**Installing the Library**

First, you need to clone my GitHub repository.
```bash
git clone https://github.com/FareedKhan-dev/Most-powerful-NLP-library.git
```
If you don’t have Git installed on your machine, you can download the repository as a ZIP file.

![[download repository as zip file from github link](https://github.com/FareedKhan-dev/Most-powerful-NLP-library)](https://cdn-images-1.medium.com/max/4676/1*82qE07mfI1EUX38ZkuaVzA.png)

Once you have cloned the repository, you need to install the required dependencies that allow you to work with the Gemini API.
```bash
# Install the Google Generative AI library
pip install -q -U google-generativeai
```

**Initiating the Library**

In the previous step, we installed the required dependencies and cloned our NLP library. Now, we need to import the necessary library that will fetch Gemini LLM API calls and instantiate the required API key.
```python    
# Import the Google Generative AI library
import google.generativeai as genai

# Initialize the GenerativeModel with 'gemini-pro'
model = genai.GenerativeModel('gemini-pro')

# Configure the library with your API key
genai.configure(api_key="Your-API-key")
```

You can obtain your API key from [here](https://makersuite.google.com/app/apikey). Once you have the key, proceed to the next step.



**Tokenization Example**


```python
from pre_processing import tokenize_text

user_input = '''The cats are running and playing in the gardens, while the dogs are barking loudly and chasing their tails'''

my_output = tokenize_text(user_input, model)

print(type(my_output), my_output)
```

    <class 'list'> ['The', 'cats', 'are', 'running', 'and', 'playing', 'in', 'the', 'gardens', ',', 'while', 'the', 'dogs', 'are', 'barking', 'loudly', 'and', 'chasing', 'their', 'tails']
    

**Lemmatization Example**


```python
from pre_processing import lemmatize_text

# Assuming 'your_model' is the instance of your model
user_input = '''The cats are running and playing in the gardens, while the dogs are barking loudly and chasing their tails'''
lemmatized_sentence = lemmatize_text(user_input, model)
print(lemmatized_sentence)
```

    The cat be run and play in the garden, while the dog be bark loud and chase their tail
    

**Stemming Example**


```python
from pre_processing import stem_text

user_input = '''The cats are running and playing in the gardens, while the dogs are barking loudly and chasing their tails'''

stemmed_sentence = stem_text(user_input, model)

print(stemmed_sentence)
```

    the cat ar run and play in the garden, whil the dog ar bark loud and chas their tail
    

**Pattern Matching Example**


```python
from pre_processing import extract_patterns

user_input = '''The phone number of fareed khan is 123-456-7890 and 523-456-7892. Please call for assistance and email me at x123@gmail.com'''

# You can add more patterns here separated by commas
pattern_matching = '''emal, phone number, name'''

extracted_patterns = extract_patterns(user_input, pattern_matching, model)

print(extracted_patterns)
```

    ['123-456-7890', '523-456-7892', 'x123@gmail.com', 'fareed khan']
    

**Text Cleaning Example**


```python
from pre_processing import clean_text

user_input = '''faree$$@$%d khan will arrive at 9:00 AM. He will@%$ 1meet you at the airport. He will be driving a black BMW. His license plate is 123-456-7890.'''

cleaned_text = clean_text(user_input, model)

print(cleaned_text)

```

    Fareed Khan will arrive at 9:00 AM. He will meet you at the airport. He will be driving a black BMW. His license plate is 123-456-7890.
    

**HTML tags removal Example**

```python
from pre_processing import remove_html_tags

user_input = '''<p>This is <b>bold</b> and <i>italic</i> text.</p>'''

# You can add more tags here separated by commas
html_tags = '''<p>, <b>, <i>''' 

cleaned_text = remove_html_tags(user_input, html_tags, model)

print(cleaned_text)

```

    This is bold and italic text.
    

**Replace text Example**


```python
from pre_processing import replace_text

user_input = '''I like cats, but I don't like dogs.'''

# You can add more rules here separated by commas
replacement_rules = '''all animals to rabbits'''

modified_text = replace_text(user_input, replacement_rules, model)

print(modified_text)
```

    I like rabbits, but I don't like dogs.
    

**Generate Embedding Vectors Example**


```python
from pre_processing import extract_embeddings

user_input = ["cats are running and playing in the gardens", "dogs are barking loudly and chasing their tails"]

# extract_embeddings() takes a list of strings as input
modified_text = extract_embeddings(user_input)

# print first 10 values of embedding vector the first sentence
modified_text['embedding'][0][:10]
```




    [0.0195884,
     0.024218114,
     -0.029704109,
     -0.05665759,
     -0.011961627,
     -0.026998892,
     -0.024396203,
     -0.021466378,
     0.021265924,
     -0.0027763597]



**Text Classification:**
   - Sentiment Analysis
   - Topic Classification
   - Spam Detection


```python
from core_nlp import analyze_sentiment

user_input = "I love to play football, but today I am feeling very sad. I do not want to play football today."

# You can add more categories here separated by commas (Default: positive, negative, neutral)
category = "positive, negative, neutral"

sentiment_result = analyze_sentiment(input_text=user_input, category=category, explanation=True, model=model)
print(sentiment_result)
```

    **Category: Negative**
    
    **Short Explanation:**
    
    The overall sentiment of the text is negative. The author expresses a love for football but then goes on to say that they are feeling very sad and do not want to play football today. This indicates a negative sentiment towards the activity of playing football.
    


```python
from core_nlp import classify_topic

user_input = "I love to play football, but today I am feeling very sad. I do not want to play football today."

# You can add more topics here separated by commas (Default: story, horror, comedy)
topics = "topics are: story, horror, comedy"

topic_result = classify_topic(input_text=user_input, topics=topics, explanation=True, model=model)

print(topic_result)

```

    Topic: Story
    Explanation: The input text is a story about a person who loves to play football but is feeling sad and does not want to play today. The text does not contain any elements of horror or comedy, so the topic is classified as "story".
    

**Spam Detection Example**


```python
from core_nlp import detect_spam

user_input = "you have just won $14000, claim this award here at this link."

# You can add more categories here separated by commas (Default: spam, not spam, unknown)
category = 'spam, not_spam, unknown'

spam_result = detect_spam(input_text=user_input, category=category, explanation=True, model=model)

print(spam_result)
```

    spam
    
    Explanation: The message contains the promise of a large monetary reward, which is a classic tactic used by spammers to attract attention and entice people to click on the link. The use of the word "claim" also indicates the sender's desire to obtain personal information from the recipient, which is another common goal of spammers.
    

**NER Detection Example**


```python
from core_nlp import detect_ner

user_input = "I will meet you at the airport sharp 12:00 AM."

# You can add more categories here separated by commas (Default: erson, location, date, number ... cardinal)
ner_tags = 'person, location, date, number, organization, time, money, percent, facility, product, event, language, law, ordinal, misc, quantity, cardinal'

ner_result = detect_ner(input_text=user_input, ner_tags=ner_tags, model=model)
print(ner_result)

```

    airport: facility
    12:00 AM: time
    

**POS Tagging Example**


```python
from core_nlp import detect_pos

user_input = "I will meet you at the airport sharp 12:00 AM."

# you can add more categories here separated by commas (Default: NOUN, 'noun, verb, ..., cashtag_phrase, entity_phrase')
pos_tags = 'noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, determiner, cardinal, foreign, number, date, time, ordinal, money, percent, symbol, punctuation, emoticon, hashtag, email, url, mention, phone, ip, cashtag, entity, noun_phrase, verb_phrase, adjective_phrase, adverb_phrase, pronoun_phrase, preposition_phrase, conjunction_phrase, interjection_phrase, determiner_phrase, cardinal_phrase, foreign_phrase, number_phrase, date_phrase, time_phrase, ordinal_phrase, money_phrase, percent_phrase, symbol_phrase, punctuation_phrase, emoticon_phrase, hashtag_phrase, email_phrase, url_phrase, mention_phrase, phone_phrase, ip_phrase, cashtag_phrase, entity_phrase'

pos_result = detect_pos(input_text=user_input, pos_tags=pos_tags, model=model)

print(pos_result)

```

    I: pronoun
    will: verb
    meet: verb
    you: pronoun
    at: preposition
    the: determiner
    airport: noun
    sharp: adverb
    12:00: time
    AM: time
    .: punctuation
    

**Machine Translation Example**


```python
from core_nlp import translate_text

user_input = "I will meet you at the airport sharp 12:00 AM."

source_language = "english"

target_language = "spanish"

translation_result = translate_text(user_input, source_language, target_language, model)

print(translation_result)
```

    Te encontraré en el aeropuerto en punto de las 12:00 AM.
    

**Text Summarization Example**


```python
from core_nlp import summarize_text

user_input = "I will meet you at the airport sharp 12:00 AM."

summary_length = "medium" # short, medium, long

summary_result = summarize_text(user_input, summary_length, model)

print(summary_result)

```

    You are requested to meet at the airport promptly at midnight.
    

**Question Answering Example**


```python
from core_nlp import answer_question

question_text = "Is it possible that an ant can kill a lion?"

answer_result = answer_question(question_text, model=model)

print(answer_result)
```

    No, it is not possible for an ant to kill a lion.
    

**Text Generation Example**


```python
from core_nlp import generate_text

prompt_text = "poem on a friendship between a cat and a mouse"

generation_length = "short"

generated_text = generate_text(prompt_text, generation_length, model)

print(generated_text)
```

    In a tale of unique bond, so true,
    A cat and a mouse, friendship grew.
    Amidst the world of chase and prey,
    Their hearts entwined in a different way.
    
    The cat, playful and sleek and sly,
    The mouse, nimble and bright of eye,
    Met one day in the corner old,
    Where stories and secrets were untold.
    
    They talked and laughed, they shared their dreams,
    Of chasing stars and moonbeams.
    No longer bound by predator or prey,
    They found a bond that would never sway.
    
    Together they'd explore the night,
    Underneath the silver moonlight.
    A dance of shadows, soft and sweet,
    Where differences were obsolete.
    
    They'd share their meals, they'd share their home,
    A friendship that would forever roam.
    In a world of chaos, a gentle grace,
    A cat and a mouse, in harmony's embrace.
    

**Semantic Role Labeling (SRL) Example**


```python
from core_nlp import perform_srl

user_input = "tornado is approaching the city, please take shelter"

srl_result = perform_srl(user_input, model)

print(srl_result)
```

    Predicate: approach
    Roles:
    - Agent: tornado
    - Theme: city
    

**Intent Recognition Example**


```python
from core_nlp import recognize_intent

user_input = "tornado is approaching the city, please take shelter"

intent_result = recognize_intent(user_input, model)

print(intent_result)
```

    Intent: Emergency alert
    

**Paraphrasing Detection Example**


```python
from core_nlp import paraphrasing_detection

user_input = ['''The sun sets in the west every evening.''','''Every evening, the sun goes down in the west.''']

intent_result = paraphrasing_detection(input_text=user_input, explanation=True, model=model)

print(intent_result)
```

    yes
    Both sentences express the same idea that the sun sets in the west every evening. They use different words to convey the same meaning, such as "sets" and "goes down" for the verb and "every evening" for the temporal modifier.
    
## Handling Large Data

Up until now, we’ve worked with relatively small text data, like short sentences. If you need to handle larger text, while I haven’t implemented it yet, one approach is to break your text data into chunks and process it accordingly. Here’s an example of how to work with a bigger dataset.

    # Example text dataset
    text_dataset = "some_big_text_file.txt"
    
    # Break the text into sentences based on full stops
    sentences = text_dataset.split('. ')
    
    # some ner_tags you have defined
    ner_tags = "person, organization ..."
    
    # Applying NER on it
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}:")
        
        # Applying NER on each sentence
        detect_ner(input_text=sentence, ner_tags=ner_tags, model=model)

Another approach to handling larger data is to break it into more extensive chunks, for example, 500 sentences per chunk, to preserve dataset information. If you want to apply the text_summarization task, you can then provide the summaries of each chunk in a combined manner to generate one detailed summary for the entire text.

![Visual Illustration of how to handle large text data](https://cdn-images-1.medium.com/max/3576/1*Wf17hrjz62k5xe6h3As4Jg.png)

There are several ways to handle big data, but the approaches I’ve just shared are among the most common and practical.

**Customizing the Library**

Customizing the library involves including your own functions, and a well-crafted prompt is essential for making your customized functions work. For instance, if you want to create a paraphrasing-checking function, you need to start with a prompt for the paraphrasing task.
```python
# Question to be asked for determining paraphrasing
question = f'''Given the input text, determine if two sentences are paraphrases of each other.
Sentence 1: {user_input[0]}
Sentence 2: {user_input[1]}
Answer must be 'yes' or 'no'.
{explanation}
'''
```

When creating a customized function, it’s crucial to explain the expected output from Gemini to maintain consistency across runs. Additionally, defining the answer format is essential; for instance, in tokenization, you may specify that the output format should be a list. To achieve this, you can later convert the string representation to an actual list using the ast Python library. In the paraphrasing task, the rest of the prompt remains relatively constant, with changes depending on how many sentences you want to input—I've considered two in this example.

Once you create your prompt, you can build a function on top of it.
```python
# function for paraphrase detection
def paraphrasing_detection(input_text, explanation, model):

    # Check if explanation is required
    explanation_text = 'short explanation: ' if explanation else 'no explanation'

    # Question to be asked for determining paraphrasing
    question = f'''Given the input text, determine if two sentences are paraphrases of each other.
    Sentence 1: {input_text[0]}
    Sentence 2: {input_text[1]}
    Answer must be 'yes' or 'no'.
    {explanation_text}
    '''

    # Generate response
    response = model.generate_content(question)
    return response.text.strip()
```
You can easily call that function on top of your text data.
```pytho
# Import the paraphrasing_detection function from the core_nlp module
from core_nlp import paraphrasing_detection

# User input text
user_input = ['''The sun sets in the west every evening.''', '''Every evening, the sun goes down in the west.''']

# Perform paraphrasing detection using the specified model
intent_result = paraphrasing_detection(user_input, explanation=True, model=model)

# Print the paraphrasing detection result
print(intent_result)
```

```bash
##### OUTPUT OF ABOVE CODE #####

Answer: yes
Short Explanation: Both sentences express the same idea that the sun 
sets in the west  every evening. They use different words to convey 
the same meaning,  such as "sets" and "goes down" for the verb and 
"every evening" for temporal modifier.

##### OUTPUT OF ABOVE CODE #####
```

**What’s Next**

There are many more features introduced in this library. This is just a glimpse of how LLMs reshape NLP tasks and simplify the handling of text data. Explore the full potential by checking out my [GitHub repository](https://github.com/FareedKhan-dev/Most-powerful-NLP-library), which includes features like generating embeddings for cosine similarity, text summarization, and more. Feel free to adapt the library for your specific domain, whether it’s medical or any other. I hope you enjoy reading this blog.

If you want to build your own LLM from scratch or understand the mathematical aspects of transformers, you can refer to my other blogs:
[**Solving Transformer by Hand: A Step-by-Step Math Example**
*Performing numerous matrix multiplications to solve the encoder and decoder parts of the transformer*levelup.gitconnected.com](https://levelup.gitconnected.com/understanding-transformers-from-start-to-end-a-step-by-step-math-example-16d4e64e6eb1)
[**Building a Million-Parameter LLM from Scratch Using Python**
*A Step-by-Step Guide to Replicating LLaMA Architecture*levelup.gitconnected.com](https://levelup.gitconnected.com/building-a-million-parameter-llm-from-scratch-using-python-f612398f06c2)

<hr>

**License**
MIT License
