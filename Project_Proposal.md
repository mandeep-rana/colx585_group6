### Project Proposal

##### Introduction
 - The task we have chosen to work on is Automatic Speech Recognition with Wav2Vec2 on Spanish Audio to English text. 
 - Our initial plan is to utilize a fine tuned XLSR-Wav2Vec2 Multilingual model using Huggingface Transformer, input the Spanish audio dataset from openslr.org or commonvoice.mozilla.org. The overall model architecture will be pipeline based, where audio input will first be converted into text using XSLR-Wav2Vec2. Then we will use this text output from XLSR as an input for an NMT model, which will translate the Spanish text to English text-based translation.

##### Motivation
- It's a very intriguing, and exciting project to work on, moreover, it is a great opportunity for us to learn new models which fall into the speech data category (example: XLSR), and combine these new tools with the stuff we learned in the previous block (NMT). Hopefully, we aim to create something which is very unique, and is relatively new in terms of work done in the industry.
- We hope to create this project in State of the art fashion, and hopefully manage to well integrate the pipeline. Hopefully created a system which can be beneficial in the industry or at least try to explore this domain a little further than we have previously.

##### Motivation
Group 6 will use Google Colab for training the model, and Group 6 will use PyTorch as the framework. The codebase will be based on "Fine-tuning XLSR-Wav2Vec2 for Multi-Lingual ASR with 🤗 Transformers" by Patrick von Platen, which is avaliable at (https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb)



##### Conclusion
- We hope to deliver a very efficient pipeline system which corresponds well to the task defined above, in a timely manner. 