##### Abstract
This project presents a Portuguese speech to English text pipeline model. Since Portuguese is a low-resource language, a quick look at its applications in the field of ASR shows that most of the work done is monolingual (e.g. Portuguese speech to Portuguese text tasks). Furthermore, with the introduction of XLSR Wav2Vec2 in Sep 2020, a more powerful ASR model can be built using low-resource languages and applied to multilingual tasks. With this, the group builds on XLSR Wav2Vec2 model, fine-tuning it on Portuguese data to generate Portuguese texts that are given to an OpenNMT transformer model to translate it to English. Our model yields a 7.31 BLEU score and Word Error Rate (WER) of 0.2860 which are reasonable in a low-resource setting.

##### Introduction
The field of Automatic Speech Recognition (ASR) has been around for more than 60 years gradually being developed on high-resource languages like English and then moving on to more language-specific systems that include low-resource languages like Portuguese. Currently, ASR is used as a tool in different fields to help treat speech dysfunctions, evaluate second-language proficiency, improve virtual assistant programs, and develop automatic transcription systems for movies and TV shows (De Lima & Da Costa-Abreu, 2020). These ASR applications are mostly monolingual but can be extended to multilingual settings when combined with Machine Translation. Group-6 aims to contribute to the growing body of multilingual ASR applications on Portuguese using a pipeline model that generates English text from Portuguese audio.

A summary of related works focusing on Portuguese ASR is provided below showing its progress throughout the years, followed by a description of the data used by the project to train the model. Then, an explanation of the methods and a description of the pipeline model is given as well as the different experiments that ran to accomplish the speech-to-text task. Finally, a table summarizing the results of the project is provided with some concluding remarks.
 
##### Related Works
A survey of previous literature shows that there is limited but fast-growing research on Portuguese ASR. Among these is Fuzzy Inference System and Genetic Algorithm model used by Silva and Serra (2014) which is subsequently followed by different methods such as the use of Support Vector Machines and Convolutional Neural Networks to improve model performance with noisy audio (Silva and Barbosa, 2015; Santos et al., 2015). Since Portuguese is a low-resource language, researchers found it difficult to train monolingual models such as those mentioned above. Because of this, some researchers have switched to a cross-lingual approach to ASR which has outperformed the classic monolingual approach (Ghoshal et al., 2013) and have developed better, more complex Deep and Convolutional Neural Networks for low-resource languages including Portuguese (Shaik et al., 2015) with the most recent development introduced through Facebook's XLSR Wav2Vec2 model pre-trained in 53 languages (Conneau et al., 2020).

Despite these developments on Portuguese ASR, there is still a lack of work done on Portuguese speech to other language texts. As mentioned previously, most of the work on Portuguese ASR is used on monolingual applications (e.g. Portuguese speech to Portuguese text). In this project, we extend Portuguese ASR to a multilingual setting by translating Portuguese Speech to English text using a fine-tuned XLSR Wav2Vec model with a Machine Translation transformer model.

##### Data
The data used for this project includes a total of around 63 hours of transcribed Portuguese audio from Mozilla Commonvoice and approximately 50 hours of this data is validated. The audio files are in mp3 format with a sampling rate of 48kHz. The dataset consists of 1120 unique voices with an 81% male and 3% female split. Moreover, 67% of the audio data is from participants between 19-39 years of age, 17% is from participants between 40-59 years old, and less than 3% is from participants younger than 19 years old. For model training, the data was split into approximately 51 hours of training data, 6 hours of validation data, and 6 hours of test data(Train 80%, Dev 10%, Test 10%).

In addition to this, Group-6 also used Portuguese-English aligned text from OpenSLR. These texts come from a multilingual corpus of TEDx Talks for speech recognition and translation found at (https://www.openslr.org/100). 

##### Methods
Group-6 used Google Colab Pro for training the model, and PyTorch as the framework. The project's input is Portuguese audio, and the output is the English text. Therefore, the project is broken into two parts: the upstream task of converting Portuguese audio into vector representation and the downstream task of converting that Portuguese vector representation into English text.

For the Portuguese audio to Portuguese text task, the codebase is based on "Fine-tuning XLSR-Wav2Vec2 for Multi-Lingual ASR with Huggingface Transformers" by Patrick von Platen, which is available at (https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb). The team trained the XLSR-Wav2Vec2 model on Portuguese audio transcript dataset from Mozilla Commonvoice project.

For the XLSR-Wav2Vec2 model, we trained the model with attention dropout, hidden dropout, layer dropout all set to 0.1. We disabled feature projection dropout and set it to 0. We set the mask time probability to 0.05. We also set ctc loss reduction method to mean. For the training process, we trained the model for 30 epochs, at a learning rate of 0.00003.

For the Portuguese text to English text task, the team used OpenNMT Transformer model, with pre-trained Portuguese embeddings from FastText with a dimension of 300. We used Transformer for both the encoder and decoder. The RNN size is set to 512, and the layer is set to 6. We used 8 heads for the model. For the optimizer, we used Adam optimizer with a learning rate of 2.0, beta1 of 0.9, and beta2 of 0.998. We also set the dropout rate to 0.2 and used 0.1 for attention dropout.

##### Experiments
Group-6 conducted three full experiments and many test runs. The first experiment is to test the performance of the Portuguese audio to Portuguese text model on the baseline XLSR-Wav2Vec2 model, without fine-tuning. The baseline model generated a word error rate (WER) of 0.30 on the Mozilla Commonvoice development set data. The team then fine-tuned the baselined XLSR-Wav2Vec2 model with aligned Portuguese audio to Portuguese text data from Mozilla Commonvoice. The fine-tuned model generated a word error rate(WER) of 0.286, which is an improvement on the baseline result.

For evaluating the entire pipeline, the team used silver prediction data, generated from Google translate, because there is a lack of aligned Portuguese Audio to English Text data. There is a CoVoST2 model (from Massively Multilingual Speech-to-Text Translation by Changhan Wang, Anne Wu, Juan Pino) that can also be potentially used to generate the silver prediction data. However, the CoVoST 2 data will take too much time to generate, thus the team has to create a silver prediction dataset for the Portuguese development audio, using Google translate, and use this dataset for evaluating the model. The entire pipeline was able to achieve a BLEU score of 7.31 and a word error rate of 0.286.

##### Results
| Component | Model | Input | Output | WER | BLEU |
|-----------|-------|-------|--------|-----|------|
| Source Audio to Source Text | Wav2vec2 Baseline | Portuguese Audio|Portuguese Text | 0.30 | BLEU |
| Source Audio to Source Text | Wav2vec2 Fine-tuned | Portuguese Audio|Portuguese Text | 0.286 | 58.39 |
| Source Audio to Target Text | Transformer Fine-tuned | Portuguese Audio|English Text | 0.286 | 7.31 |

The results mentioned above are final results after rigorous training. We got these results by fine-tuning the ASR model and separately the MT model, then combining them to form a pipeline structure. Due to time and space constraints, we believe these are very good scores on a low resource language like Portuguese. We could have improved the model if we had implemented different approaches as mentioned in the future work section given additional time. As per the data size of the model and our research we found out that a WER score in the range of 25-30 is considered as a reasonable score. We could have done more hyperparameter tuning given we had better GPU machines as hyperparameter tuning took ~15-20 hours per session and Google Colab crashed frequently. 

##### Code

**ASR Model Code**
https://github.ubc.ca/jurquico/colx585_group6/blob/master/milestone_4/XLSR_Wav2Vec2_portuguese.ipynb

**MT model**
https://github.ubc.ca/jurquico/colx585_group6/blob/master/milestone_4/OpenNMT%20PT%20Text%20to%20EN%20Text.ipynb

https://github.ubc.ca/jurquico/colx585_group6/blob/master/milestone_4/transformer-fasttext-10k-vocab.yml


##### Conclusion
We set out to make a pipeline consisting of a working ASR model which converted Portuguese speech data into Portuguese text and MT model which translated Portuguese text to English text.
We can state that we have accomplished our task successfully. We have got WER score of 0.2860 and an Overall BLEU score of 7.31
##### Future Work
###### Adding Spellcheck/Language  Model
Including a spellcheck model or Language model before MT model to enhance the performance

###### Making an End-to-End model
Instead of doing a pipeline model we could have tried for tightly coupled End-To-End ST model for better results.

###### Training with High Resource Language
We could have gotten better scores if we can train on High Resource Language.

###### Reference List

Conneau, A., Baevski, A., Collobert, R., Mohamed, A., Auli, M., 2020. Unsupervised cross-lingual representation learning for speech recognition.  arXiv:2006.13979v2.https://arxiv.org/abs/2006.13979.

De Lima, T., Da Costa-Abreu, M., 2020. A survey on automatic speech recognition systems for Portuguese language and its variations. In: Computer Speech & Language(62).https://doi.org/10.1016/j.csl.2019.101055

Ghoshal, A., Swietojanski, P., Renals, S., 2013. Multilingual training of deep neural networks. In: Proceedings of the IEEE International Conference on Acoustics,Speech and Signal Processing, pp. 7319–7323.https://doi.org/10.1109/ICASSP.2013.6639084

Santos, R.M., Matos, L.N., Macedo, H.T., Montalvao, J., 2015. Speech recognition in noisy environments with convolutional neural networks. In: Proceedings of theBrazilian Conference on Intelligent Systems (BRACIS), pp. 175–179.https://doi.org/10.1109/BRACIS.2015.44.

Shaik, M.A.B., Tuske, Z., Tahir, M.A., Nußbaum-Thom, M., Schluter, R., Ney, H., 2015. Improvements in RWTH LVCSR evaluation systems for polish, portuguese, english, urdu, and arabic. In: Proceedings of the Sixteenth Annual Conference of the International Speech Communication Association.

Silva, W., Serra, G., 2014. Intelligent genetic fuzzy inference system for speech recognition: an approach from low order feature based on discrete cosine tranform. J. Control, Autom.Electr.Syst.25(6),689-698.https://doi.org/10.1007/s40313-014-0148-0.

Silva, W.L.S., Barbosa, F.G., 2015. Automatic voice recognition system based on multiple support vector machines and mel-frequency cepstral coefficients. In:Proceedings of the 11th International Conference on Natural Computation (ICNC), pp. 665–670.https://doi.org/10.1109/ICNC.2015.7378069.
