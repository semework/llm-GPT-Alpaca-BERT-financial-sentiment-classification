# llm-GPT-Alpaca-BERT-financial-sentiment-classification &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

 <a href="https://www.apache.org/licenses/LICENSE-2.0"> <img src="https://www.apache.org/img/asf-estd-1999-logo.jpg" alt="seaborn" style="height:20px; "></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://tlo.mit.edu/learn-about-intellectual-property/software-and-open-source-licensing"> <img src="https://tlo.mit.edu/sites/default/files/MIT-Sterol-History_0.jpg" alt="MIT License" style="height:20px; "></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://github.com/langchain-ai"><img src="https://avatars.githubusercontent.com/u/126733545?s=200&v=4" alt="langchain" style="height:20px; "></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://badge.fury.io/py/alpaca-trade-api"> <img src="https://badge.fury.io/py/alpaca-trade-api.svg" alt="alpaca-trade-api" style="height:20px; "></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://circleci.com/gh/alpacahq/alpaca-trade-api-python"> <img src="https://circleci.com/gh/alpacahq/alpaca-trade-api-python.svg" alt="circleci" style="height:20px; "></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://pyup.io/repos/github/alpacahq/alpaca-trade-api-python/"> <img src="https://pyup.io/repos/github/alpacahq/alpaca-trade-api-python/shield.svg" alt="circleci" style="height:20px; "></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://pypi.org/project/openai/"> <img src="https://downloads.intercomcdn.com/i/o/406339/b83d08f20c6d78f99331343f/71bc35f3a54fb38a5fff9ef2d4c1879d.png" alt="langchain" style="height:20px; ">
 
<div style="text-align:center;color:white">
    <h2> Large Language Model (LLM) sentiment analysis of financial news using Alpaca, BERT and ChatGPT</h2>
</div>

### *An experiment  with the fascinating potential of large language models to efficiently classify short news summaries and headlines into 'positive', 'neutral' and 'negative' sentiments. Here we use Alpaca, BERT and ChatGPT to:*
1. import news headings and summaries for specific stock symbols
2. use the language (NLP) model Bidirectional Encoder Representations from Transformers (BERT) to tokenize and classify the news data into sentiments 
3. classify the same news data using OpenAI's gpt-3.5 to do the same 

### About this repo   
#### *A jupyter notebook (```llm_sentiment_classifier.ipynb```) which takes in financial news for ticker (AKA stock symbols) and returns sentiments is provided. Final results should look something like this:*
<img src="https://github.com/semework/llm-GPT-Alpaca-BERT-financial-sentiment-classification/blob/main/assets/images/semtiments.png" 
 style="display: block;
  margin-left: auto;
  margin-right: auto;
  height: 25%;" /> 
 
### Approach steps:

#### Step 0 - Install the necessary libraries - uncomment and run the following cells if they are not already installed. This is followed by importing the necessary libraries. # Langchain sometimes has issues with dependencies and it is recommended to install with upgrade

```commandline
pip -q install alpaca-trade-api alpaca-py transformers openai tiktoken
pip -q install langchain --upgrade
```

<div style="text-align:center;color:red">
    <h2>1 - Import keys saved in your environment</h2>
</div>

#### Most APIs provide a security option to ensure that you can store your authentication details in environment variables. This enables you:
1. to authenticate your logins and code tracking in code depos such as GitHub
2. in addition to #1, to control your privileges such as OpenAI's tokens
3. to protect yourself from inadvertently exposing your secret IDs and keys in any environment such as live trading platforms such as Alpaca
#### For best practices read:
1. OpenAI's advice: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
1. Alapaca: https://medium.com/software-engineering-learnings/algorithmic-trading-with-alpaca-and-python-c81bad480053
3. General: https://www.twilio.com/blog/how-to-set-environment-variables-html

```commandline
openai.api_key = os.environ["OPENAI_API_KEY"]
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_KEY_SECRET")
print('keys imported')
```

---------- 
<div style="text-align:center;color:red">
    <h2> 2 - Setup your llm model and construct a pydantic base model</h2>
</div>

##### Pydantic defines objects via models (classes which inherit from pydantic.BaseModel).
More info here: https://docs.pydantic.dev/latest/usage/models/ which states:
- These models are similar to Python's dataclasses with some differences that streamline certain workflows related to validation, serialization, and JSON schema generation. Untrusted data can be passed to a model and, after parsing and validation, Pydantic guarantees that the fields of the resultant model instance will conform to the field types defined on the model.

---------- 
<div style="text-align:center;color:red">
    <h2> 3 - Prepare sentiment analysis models and pipeline</h2>
</div>

#### Approach and cautionary note:
- we will use a fine-tuned (trained for financial news) Hugging Face model (BERT) to analyze the article's headline and summary sentiment
- initial model downloads might take some time
#### Next, create an alpaca client, choose stock tickers and decide how many days of news to scrape

---------- 
<div style="text-align:center;color:red">
    <h2> 4 - Create a functions to classify individual news and calculate an average sentiment per ticker and to process all tickers</h2>
</div>

#### Important considerations:
- more recent news might be more relevant
- the sentiment confidence also gives us a clue how certain the algorithm is about its classification

#### Approach:
- weigh recent news more heavily (straightforward linear increase, going from old to new - although there can be many variations of this approach such as inverted/hyperbolic, linear-with-noise, etc. approaches)
- use sentiment confidence to adjust our weights. i.e. multiply recency score with the score
- the function ```sentiment_to_weighed``` takes care of the weighing
- the function ```sentiment_analysis``` takes in a list and tickers and returns a weighed sentiment per ticker
- since OpenAI's tocken allotments deplete quickly, a few lines in  ```sentiment_analysis``` are commented out and a flag is given. Uncomment them if you have enough tokens left (after changing ```do_llm``` flag to "1". Note that llm sentiment classification is slow.

### What you should see for each ticker:

<img src="https://github.com/semework/llm-GPT-Alpaca-BERT-financial-sentiment-classification/blob/main/assets/images/goog_sentiments.png" style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 75%;"/> 

## Contributing and Permissions

Please do not directly copy anything without my concent. Feel free to reach out to me at https://www.linkedin.com/in/mulugeta-semework-abebe/ for ways to collaborate or use some components.

## License

langchain under MIT and Alpaca trade api under Apache License 2.0. Please view [LICENSE](https://tlo.mit.edu/learn-about-intellectual-property/software-and-open-source-licensing) and (https://www.apache.org/licenses/LICENSE-2.0) for more details. For other packages click on corresponding links at the top of this page (first line).
