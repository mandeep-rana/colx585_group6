


# Milestone 1 Feedback:

### Like:

- Spoken Language Translation is extremely ambituous and will allow you to get your hands dirty on many different aspects of speech processing and computational linguistics, I am really excited to see how you do!
- Well laid out proposal covers the main points in an easy to read format.
- Good data selection for Speech portion of the pipeline.
- You've really thought through the data storage part of the issue (w.r.t. google drive and colab).
- Thorough team contract!
- Like that you've already looked at the Huggingface(VonPlaten) wav2vec tutorials.

### Questions:

- Will you be using a language model at the end of your pipeline?
- Do these corpora have train, dev, test splits?
- How will you downsample your audio?
- Have you listened to the audio to determine the audio quality? (is it noisy?)
- How will you know when you have fine-tuned your ASR model enough?
- How are you pre-processing the transcriptions?
- Do you have a baseline model or comparison?


### Recommended Next Steps / Direct Feedback:  

- Overal metric for SLT is BLEU. So you'll report WER on ASR model, BLEU on MT (test set of whatever corpus you use to train the MT), and then BLEU on the entire pipeline.
- Make sure to indicate hrs of different types of data.
- Wav2Vec2 isn't 'just' a transformer model, make sure you are situtating it appropriate (among other unsupervised/self-supervised speech processing models such as wav2vec, vq-wav2vec, DeCoAR, DeCoAR 2, mockingjay etc.)
- "End-to-End" has a really specific meaning (especially with respect to SLT). This is not an "Eng-to-End" model, but rather a "Pipeline" or "Cascade" approach (e.g. it consists of more than just one model).

