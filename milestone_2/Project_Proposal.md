### Project Proposal

##### Introduction
 - This project will use `Automatic Speech Recognition(ASR)` with Wav2Vec2 on Spanish Audio and convert it to English text. 
 - Our plan is to utilize a fine tuned XLSR-Wav2Vec2 Multilingual model using `Huggingface Transformer`, input the Spanish audio dataset from `openslr.org` or `commonvoice.mozilla.org`. The overall model architecture will be pipeline based, where audio input will first be converted into Spanish text using XLSR-Wav2Vec2. Then we will use this text output from XLSR as an input for an MT model, which will translate the Spanish text to English text-based translation.

The overall goal still holds true from Milestone 1. The team is building a machine learning model for translating for either of  Estonian, Czech and Portuguese audio( depending on best WER)  into English text.


![](./img/flow.PNG)

##### Motivation
- It's a very intriguing, and exciting project to work on, moreover, it is a great opportunity for us to learn new models which fall into the speech data processing category (example: XLSR- released in September 2020), and combine these new tools with what we have learned in the previous block (NMT , FastText etc). We aim to create a new model which is relatively new in terms of work done in the industry.
- We want to design this project in State of the art fashion and create an pipeline model for audio to language translation. This model can be further extended to translate audio from one language to text of different language which will help others to explore this domain.
- By working on this project, we will learn newly introduced models and technology which will help us to understand these concepts in detail.

##### Data
We will be using multilingual corpus of ‘TEDx talks for speech recognition and translation’ from http://openslr.org/100/.  Statistics about the data is as below:

Out of scope now:

|Data|Size Information|
|---|---|
| Spanish speech and transcripts data size  | 35GB |
| Spanish speech and transcripts with aligned English translations  |  13GB |
| Talks  | 1031   |
| Sampling Rate  | 48KHZ   |
|  Total hours of audio | ~216  |
|  Gogle Drive space | 100GB  |
|  Google Colab Pro space | 190GB  |

    
| Set | Talks | Sentences | Words src | Time |
| :-----: | :-: | :-: | :-: | :-: |
| Train | 988 | 102171 | 1676862 | 212h18m21s |
| Valid | 12 | 905 | 14327 | 1h56m53s |
| Test | 301 | 1012 | 15439 | 2h4m35s |

Update from Week 2:

|Google|Size Information|
|---|---|
|  Gogle Drive space | 100GB  |
|  Google Colab Pro space | 190GB  |

|Data|Size Information|
|---|---|
| Estonian speech and transcripts data size  | 732 MB |
| Talks  | 543   |
| Audio Format | mp3 |
| Sampling Rate  | 48KHZ   |
|  Total hours of audio | ~27  |
|  Validated hours of audio | ~19  |

    
| Set | Talks | Time |
| :-----: | :-: | :-: | 
| Train | 435 |  ~21 hrs |
| Valid | 54 | ~3 hrs |
| Test | 54 | ~3 hrs |

|Data|Size Information|
|---|---|
| Portuguese speech and transcripts data size  |  2 GB |
| Talks  | 1120   |
| Audio Format | mp3 |
| Sampling Rate  | 48KHZ   |
|  Total hours of audio | ~63  |
|  Validated hours of audio | ~50  |

    
| Set | Talks | Time |
| :-----: | :-: | :-: | 
| Train | 896 |  ~51 hrs |
| Valid | 112 | ~6 hrs |
| Test | 112 | ~6 hrs |


|Data|Size Information|
|---|---|
| Czech speech and transcripts data size  | 1GB |
| Talks  | 353   |
| Audio Format | mp3 |
| Sampling Rate  | 48KHZ   |
|  Total hours of audio | ~45  |
|  Validated hours of audio | ~36  |

    
| Set | Talks | Time |
| :-----: | :-: | :-: | 
| Train | 283 |  ~36 hrs |
| Valid | 35 | ~4.5 hrs |
| Test | 35 | ~4.5 hrs |


Update from Week 2:

###### Challenges

- Drive & Colab Space and training time 

After extensive testing on the Spanish audio dataset, the team finds out that Google Colab cannot handle the size of Spanish audio training dataset even we had space Google colab Pro(200 GB) and Google Drive (200GB).  Therefore, the team is shifting focus to languages such as `Estonian, Czech and Portuguese`, which all have smaller datasets. In addition, the team switched from Open SLR to Mozilla Common Voice as the dataset provider, since Mozilla Common Voice is better integrated with Wav2vec2 model than openslr. 

Each model is taking ~15-20 hours to train which is very time consuming. 

Aside from minor changes in data. The overall goal still holds true from Milestone 1. The team is building a machine learning model for translating for either of  Estonian, Czech and Portuguese audio( depending on best WER)  into English text.

Downsampling - We will make use of the librosa library to downsample the data to 16KHZ

##### Engineering
Group-6 will use Google Colab Pro for training the model, and PyTorch as the framework. The codebase will be based on "Fine-tuning XLSR-Wav2Vec2 for Multi-Lingual ASR with Huggingface Transformers" by Patrick von Platen, which is available at (https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb)

The project's input is Spanish audio, and the output is English text. Therefore, the project can be broken into two parts: the upstream task of converting Spanish audio into vector representation and the downstream task of converting that Spanish vector representation into English text.

For the audio conversion part of the project, the team plan to use XLSR-Wav2Vec2 model. For the vector to text part of the project, the team plan to try a any  of LSTM, transformer and seq2seq models with pretrained embeddings like FastText(https://fasttext.cc/docs/en/crawl-vectors.html).

Update from week 2:

The team members have trained XLSR-Wav2Vec2 models using Estonian and Portuguese data from Mozilla Common Voice. Those two languages have a small enough training dataset to be trained within Google Collab. The team have also identified a list of resources for pre-trained word embeddings for those two languages, which would be helpful in building the translation task. The team is making good progress and constantly sharing results in the slack channel.

Audio Quality : We have listened to audio quality of few files and there was not any noise issues.

language model : We will not be using any language model to enhance our model's performace due to time constraint at the end of pipeline

Pre-processing of Transcriptions - We cleaned the data using the regex. 

Base-line model - We will using same intial hyperparameters from the wav2vec2 template to baseline the model and then will fine tune the hyperparmeter to get the target WER range.



##### Previous Works

One of our team member has extensive knowledge on working on audio codecs.  He has worked on project of converting audio to multiple formats using different sampling frequency. This will help us in understanding various nuisances related to audio quality and conversion.

Additionally, all team members have worked on machine translation project last block.  One of team members MT project was selected as top scorer in last block.

##### Evaluation
We will evaluate out system on two main metrics. One is WER(Word error rate) for audio conversion and other is BLEU score for Spanish to English text conversion.

Update from week 2:
The team trained XLSR-Wav2Vec2 models on Estonian and Portuguese and tested the WER. The WER for Estonian is 0.43 and the WER for Portuguese is better at 0.282000. 

We will start with baseline WER and will try to achieve 10% improvement on WER

We will report WER on ASR model, BLEU on MT and then BLEU on the entire pipeline.

##### Conclusion
- We will deliver a working model which will fulfill the objective of the project and completed on time, with the quality as per defined in the different milestones

##### Project Milestones

Project milestone is based on current plan as of today, there may be change in some of deliverables based on the complexity or if there is any change in plan.

![](./img/timeline.PNG)
