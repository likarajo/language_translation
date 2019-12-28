# Language Translation

Creating English-to-Spanish **language translation model** using ***neural machine translation*** with ***seq2seq architecture***.  

## Background  

* **Encoder-Decoder LSTM model** having ***seq2seq*** architecture can be used to solve ***many-to-many* sequence problems**, where both inputs and outputs are divided over multiple time-steps. <br>

* The **seq2seq architecture** is a type of *many-to-many sequence modeling*, and is commonly used for a variety of tasks:
  * Text-Summarization
  * Chatbot development
  * Conversational modeling
  * **Neural machine translation**

## Dependencies

* Tensorflow
* Keras
* Numpy
* Graphviz  

```
pip install -r requirements.txt
brew install graphviz
```

## Dataset

Corpus Link: http://www.manythings.org/anki/ -> spa-eng.zip, saved in the location *data/spa-eng/spa.txt*.  

## Architecture

* The **Neural Machine Translation** model is based on **Seq2Seq architecture**, which is an **Encoder-Decoder architecture**.  
* It consists of two layers of **Long Short Term Memory** networks:  
  * *Encoder* LSTM
    * Input = Sentence in the original language
    * Output = Sentence in the translated language, with a *start-of-sentence* token
  * *Decoder* LSTM
    * Input = Sentence in the translated language, with the *start-of-sentence* token
    * Output = Translated sentence, with an *end-of-sentence* token  

## Data preprocessing

* Input does not need to be processed.
* Two copies of the translated sentence is needed to be generated.
  * with *start-of-sentence* token
  * with *end-of-sentence* token

## Tokenization

* Divide input sentences into the corresponding list of words
* Convert the input words to integers  
* Create the word-to-index dictionary for the input
* Get the number of unique words in the input
* Get the length of the longest sentence in the input
* Divide output and internediary output sentences into the corresponding list of words
* Convert the output words to integers  
* Create the word-to-index dictionary for the output
* Get the number of words in the output
* Get the length of the longest sentence in the output

## Padding

Text sentences can be of varying length. However, the LSTM algorithm expects input instances with the same length. Therefore, we convert our input and output sentences into fixed-length vectors.

* Pad the input sentences upto the length of the longest input sentence
  * zeros are padded at the beginning and words are kept at the end as the encoder output is based on the words occurring at the end of the sentence
* Pad the output sentences upto the length of the longest output sentence  
  * zeros are padded at the end in the case of the decoder as the processing starts from the beginning of a sentence  

## Word Embeddings

Deep learning models work with numbers, therefore we need to convert our words into their corresponding numeric vector representations, *and not just their integer sequence version*. There are two main differences between single integer representation and word embeddings:

* With integer reprensentation, a word is represented only with a single integer. With vector representation a word is represented by a vector of whatever dimensions is required. Hence, word embeddings capture a lot more information about words.
* The single-integer representation doesn't capture the relationships between different words. But, word embeddings retain relationships between the words.  
<br>
We use pretrained [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings from Stanford Core NLP project, saved in the location *data/glove.6B.100d.txt*

* load the GloVe word vectors into memory
* create a dictionary where words are the keys and the corresponding vectors are values
* create a matrix where the row number will represent the integer value for the word and the columns will correspond to the dimensions of the word  

## Create the model

* Create embedding layer
* Define the decoder output
  * Each word in the output can be any of the total number of unique words in the output.
  * The length of an output sentence is the length of the longest sentence in the output.
  * For each input sentence, we need a corresponding output sentence.  
  * *Hence*, **shape of the output** = (no. of inputs, length of the output sentence, no. of words in the output)
* Create one-hot encoded output vector for the output Dense layer
  * Assign 1 to the column number that corresponds to the integer representation of the word.
* Create encoder and decoder
  * Input to the encoder will be the sentence in English
  * Output from the encoder will be the hidden state and cell state of the LSTM
  * Inputs to the decoder will be the the hidden state and cell state from the encoder
  * Output from the decoder will be the sentence with start of sentence tag appended at the beginning
* Define final output layer
  * Dense layer

### Summary of the model

* input_1 is the input placeholder for the encoder, which is embedded and passed through the encoder.
* lstm_1 layer is the encoder LSTM.  
* There are three outputs from the lstm_1 layer: the output, the hidden layer and the cell state.  
* The cell state and the hidden state are passed to the decoder.
* input_2 contains the output sentences with <sos> token appended at the start, which is embedded and passed through the decoder.
* lstm_2 layer is the decoder LSTM.  
* The output from the decoder LSTM is passed through the dense layer to make predictions.

## Modifying model for prediction

While training, we know the actual inputs to the decoder for all the output words in the sequence. The input to the decoder and output from the decoder is known and the model is trained on the basis of these inputs and outputs. But, while making actual predictions, the full output sequence is not available. The output word is predicted on the basis of the previous word, which in turn is also predicted in the previous time-step. During prediction the only word available to us is <sos>.

The model is therefore needed to be modified for making predictions so that it follows the following process:

* The encoder model remains the same.  
* In the first step, the sentence in the original language is passed through the encoder, and the hidden and the cell state is the output from the encoder.
* The hidden state and cell state of the encoder, and the <sos>, is used as input to the decoder.
* The decoder predicts a word which may or may not be true.
* In the next step, the decoder hidden state, cell state, and the predicted word from the previous step is used as input to the decoder.
* The decoder predicts the next word.
* The process continues until the <eos> token is encountered.
* All the predicted outputs from the decoder are then concatenated to form the final output sentence.

## Create the prediction model

* The encoder model remains the same.  
* In each step we need the decoder hidden and the cell states.
* In each step there is only single word in decoder input. So, decoder embedding layer needs to be modified.
* The decoder output is defined.
* The decoder output is passed through dense layer to make predictions.  

### Summary of the prediction model

* lstm_2 is the modified decoder LSTM. 
* The one word is input_5.
  * The shape of the of the input sentence is now (none,1) since there will be only one word in the decoder input. On the contrary, during training the shape of the input sentence was (None,6) since the input contained a complete sentence with a maximum length of 6.
* This is passed through the modified embedding layer.
* The hidden and cell states from the previous output are input_3 and input_4 respectively.
* The decoder accepts the embedded input_5 along with input_3 and input_4.  

## Make predictions

In the tokenization steps, we converted words to integers. So, The outputs from the decoder will also be integers. However, we want our output to be a sequence of words in the translated language. Therefore, we need to convert the integers back to words. 

* Create new dictionaries for both inputs and outputs where the keys will be the integers and the corresponding values will be the words
* Pass the padded input sequence to the encoder model and predict the hidden and cell states
* Define the `<sos>` and `<eos>` tokens
* Define the output which contains single words that are predicted.
* The `<sos>` tag is used as the first word to the decoder model.
* Define the output sentence list that will contain the predicted translation
* Execute a loop that runs for the length of the longest sentence in the output
* Predict the output and the hidden and cell states using the previous ones, and store the index of the predicted word
* If the value of the predicted index is equal to the `<eos>` token, the loop terminates
* If the value of the predicted index is greater than 0, the predicted word is retrieved using the index from the output index-to-word dictionary and appended to the output sentence list
* For the next loop cycle, update the hidden and cell states, along with the index of the predicted word to make new predictions.
* After the loop terminates, the words in the output sentence list are concatenated and the resulting string is returned as the translation.

## Test

* Randomly choose a sentence from the inputs list
* Retrieve the corresponding padded integer sequence for the sentence
* Find the translation for the same 

## Improvements

* The model is trained for 5 epochs for hardware constraints. The number of epochs can be modified (increased) to get better results. 
  * Higher number of epochs may give higher accuracy but the model may overfit. 
* To reduce overfitting, we can drop out or add more records to the training set. 
  * Here we used 20,0000 records, out of which the model is trained on 18,000 records and validated on the remaining 2,000 records.

## Conclusion

* **Neural machine translation** is an advance application of Natural Language Processing and involves a very complex architecture.
* We can perform neural machine translation via the ***seq2seq architecture***, which is in turn based on the ***encoder-decoder model***.
* The encoder and decoder are LSTM networks
* The encoder encodes input sentences while the decoder decodes the inputs and generate corresponding outputs.  

## Advantages and Disadvantages

* The **seq2seq architecture** successful when it comes to ***mapping input relations to output***.
* The vanilla seq2seq architecture is ***not capable of capturing context***, and simply learns to map standalone inputs to a standalone outputs.
  * Real-time conversations are based on context and the dialogues between two or more users are based on whatever was said in the past. Therefore, a simple encoder-decoder based seq2seq model should not be used if you want to create a fairly advanced chatbot. 