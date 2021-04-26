##### Abstract
This project presents a Portuguese speech to English text cascade model. Since Portuguese is a low-resource language, a quick look at its applications in the field of ASR shows that most of the work done is monolingual (e.g. Portuguese speech to Portuguese text tasks). Furthermore, with the introduction of XLSR Wav2Vec2, a more powerful ASR model can be built using low-resource languages and applied on multilingual tasks. With this, the group builds on XLSR Wav2Vec2, fine-tuning it on Portuguese data to generate Portuguese texts that are given to an OpenNMT transformer model to translate to English. Our model yields a 7.31 BLEU score and Word Error Rate (WER) of 0.2860 which are reasonable in a low-resource setting.

##### Introduction
The field of Automatic Speech Recognition (ASR) has been around for more than 60 years gradually being developed on high-resource languages like English and then moving on to more language-specific systems that include low-resource languages like Portuguese. Currently, ASR is used as a tool in different fields to help treat speech dysfunctions, evaluate second-language proficiency, improve virtual assistant programs, and develop automatic transcription systems for movies and TV shows (De Lima & Da Costa-Abreu, 2019). These ASR applications are mostly monolingual but can be extended to multilingual settings when combined with Machine Translation. Group-6 aims to contribute to the growing body of multilingual ASR applications on Portuguese using a cascade model that generates English text from Portuguese audio.

A summary of related works focusing on Portuguese ASR is provided below showing its progress throughout the years, followed by a description of the data used by the project to train the model. Then, an explanation of the methods and a description of the cascade model is given as well as the different experiments that were ran to accomplish the speech-to-text task. Finally, a table summarizing the results of the project is provided with some concluding remarks.
 
##### Related Works
A survey of previous literature shows that there is a limited but fast-growing research on Portuguese ASR. Among these is a Fuzzy Inference System and Genetic Algorithm model used by Silva and Serra (2014) which is subsequently followed by different methods such as the use of Support Vector Machines and Convolutional Neural Networks to improve model performance with noisy audio (Silva and Barbosa, 2015; Santos et al., 2015). Since Portuguese is a low-resource language, researchers found it difficult to train monolingual models such those mentioned above. Because of this, some researchers have switched to a cross-lingual approach to ASR which has outperformed the classic monolingual approach (Ghoshal et al., 2013) and have developed better, more complex Deep and Convolutional Neural Networks for low-resource languages including Portuguese (Shaik et al., 2015) with the most recent development introduced through Facebook's XLSR Wav2Vec2 model pretrained in 53 languages (Conneau et al., 2020).

Despite these developments on Portuguese ASR, there is still a lack of work done on Portuguese speech to other language text. As mentioned previously, most of the work on Portuguese ASR is used on monolingual applications (e.g. Portuguese speech to Portuguese text). In this project, we extend Portuguese ASR to a multilingual setting by translating Portuguese Speech to English text using a fine-tuned XLSR Wav2Vec model with a Machine Translation transformer model.

##### Data
The data used for this project include a total of around 63 hours of transcribed Portuguese audio from Mozilla Commonvoice and approximately 50 hours of this data is validated. The audio files are in mp3 format with a sampling rate of 48kHz. The dataset consists of 1120 unique voices with an 81% male and 3% female split. Moreover, 67% of the audio data is from participants between 19-39 years of age, 17% is from participants between 40-59 years old, and less than 3% is from participants younger than 19 years old. For model training, the data was split into approximately 51 hours of training data, 6 hours of validation data, and 6 hours of test data.

In addition to this, Group-6 also used Portuguese-English aligned text from OpenSLR. These texts come from a multilingual corpus of TEDx Talks for speech recognition and translation found at (https://www.openslr.org/100). 

##### Methods
Group-6 used Google Colab Pro for training the model, and PyTorch as the framework. The project's input is Portuguese audio, and the output is the English text. Therefore, the project is broken into two parts: the upstream task of converting Portuguese audio into vector representation and the downstream task of converting that Portuguese vector representation into English text.

For the Portuguese audio to Portuguese text task, the codebase is based on "Fine-tuning XLSR-Wav2Vec2 for Multi-Lingual ASR with Huggingface Transformers" by Patrick von Platen, which is available at (https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb). The team trained the XLSR-Wav2Vec2 model on Portuguese audio transcript dataset from Mozilla Commonvoice project.

For the XLSR-Wav2Vec2 model, we trained the model with attention dropout, hidden dropout, layer dropout all set to 0.1. We disabled feature projection dropout and set it to 0. We set the mask time probability to 0.05. We also set ctc loss reduction method to mean. For the training process, we trained the model for 30 epochs, at a learning rate of 0.00003.

For the Portuguese text to English text task, the team used OpenNMT Transformer model, with pre-trained Portuguese embeddings from FastText with a dimension of 300. We used Transformer for both the encoder and decoder. The rnn size is set to 512, and the layer is set to 6. We used 8 heads for the model. For the optimizer, we used Adam optimizer with a learning rate of 2.0, beta1 of 0.9, and beta2 of 0.998. We also set the dropout rate to 0.2, and used 0.1 for attention dropout.

##### Experiments
Group-6 conducted three experiments. The first experiment is to test the performance of the Portuguese audio to Portuguese text model on the out-of-box XLSR-Wav2Vec2 model, without fine-tuning. The out-of-box model generated a word error rate of 0.3 on the Mozilla Commonvoice development set data. The team then fine-tuned the out-of-box XLSR-Wav2Vec2 model with aligned Portuguese audio to Portuguese text data from Mozilla Commonvoice. The fine-tuned model generated a word error rate of 0.286, which is an improvement on the out-of-box result.

For evaluating the entire pipeline, the team used a silver prediction data, generated from Google translate, because there is lack of aligned Portuguese Audio to English Text data. There is a CoVoST 2 model (from Massively Multilingual Speech-to-Text Translation by Changhan Wang, Anne Wu, Juan Pino) that can also be potentially used to generate the silver prediction data. However, the CoVoST 2 data will take too much time to generate, thus the team has to create a silver predition dataset for the Portuguese development audio, using Google translate, and use this dataset for evaluating the model. The entire pipeline is able to achieve a BLEU of 9.0 and a word error rate of 0.286.

##### Results
|Component|Model|Input|Output|WER|BLEU|
|Source Audio to Source Text|Wav2vec2 Out-of-box|Portuguese Audio|Portuguese Text|0.3|BLEU|
|Source Audio to Source Text|Wav2vec2 Fine-tuned|Portuguese Audio|Portuguese Text|0.286|58.39|
|Source Audio to Target Text|Transformer Fine-tuned|Portuguese Audio|English Text|0.286|7.31|

The BLEU score of 7.31 and WER score of 0.2860 are reasonable scores on a Low Resource language like Portuguese with very few validated hours.  We have used Silver prediction list to get the BLEU scores. We used en_silver file from the Google translate instead of CoVoST 2 due to time restrictions.

##### Conclusion
We set out to make a pipeline consisting of  working ASR model which converted Portuguese speech data into Portuguese text and MT model which translated Portuguese text to English text.
We can state that we have accomplished our task successfully. We have got WER score of 0.2860 and Overall BLEU score of 7.31

###### Adding Spellcheck/Language  Model
Including a spellcheck model or Language model before MT model to enhance the performance

###### Making an End-to-End model
Instead of doing pipeline model we could have tried for tightly coupled End-To-End ST model for better results.

###### Training with High Resource Language
We could have gotten better scores if we can train on High Resource Language.

###### Reference List