# Embedding functions

Using the appropriate model, anything can be embedded.
Different models result in different latent spaces and different embedding dimensions. Things that are similar in one embedding model can be very far from each other in the latent space created by another embedding model.
For this exercise we will focus on 1 example of turning text into vectors and see how that faciliates retrieval.

## Turning text into vectors

For this exercise we're going to use embeddings to do semantic search Rick and Morty quotes

### Installation

Installing Sentence Transformer can be done via `pip install sentence-transformers`

### Write function to generate embeddings from text

Write a function that turns text into embeddings using Sentence Transformers.

**HINT**
1. Choose a [pre-trained model](https://www.sbert.net/docs/pretrained_models.html), you don't need to create your own
2. See the API documentation and examples for Sentence Transformers to see how to encode text


### Read data

Write a function let's read the quotes from the included text file (source: https://parade.com/tv/rick-and-morty-quotes). 

### Let's put it together

Now let's build a retrieval example together using the embeddings you just generated.

Vector search is useful for retrieving data that's not part of the model's training data.

For example, if we asked the following question to ChatGPT, we get some generic sounding answer wrapping around the core "she does not make any statements about causing her parents misery".

But if we're able to use the specific quotes we have, we would get a different answer.

Let's write the following code:
1. A snippet to retrieve the quotes that are most relevant to the question.
2. Prompt engineering to instruct ChatGPT to answer based on those quotes instead.
