

# Milestone 2 Feedback:

### Like:
Love that you did an audio check! It's so simple but, extremely important that your audio is useable (not all audio corpora are as 'clean', babel and FISCHER spanish for instance are both insanely noise, which makes them bad candidates for some things).

Engineering parts look good! Model training looks great! Good call on switching to smaller training data size.

You are making really great progress!

### Questions:
Which config settings for the fine tuning are you using? (There is some discrepancy between the the huggingface defaults and the fairseq default configs, might be good to just be specific about all the training information).
While Language Models are not easy to integrate in huggingface implementation, have you considered using something like a "spell check" model (a simple seq2seq model mapping from ASR outputs to the ASR gold labels, trained on the training data (ASR outputs and their transcriptions)?
Is this still Spanish-En or have you shifted to another language? (got it... you explain this in the challenges section. I would suggest update the introduction to make it seem that 'this was the plan all along to do estonian or czech or whichever language you decide'  
You are picking based on best WER, but are you considering also how easy it will be to acquire parallel data for a MT model to go alongside it? 



### Recommended Next Steps / Direct Feedback:  
Re-write so that things are reflective of current state of things (e.g. get rid of all the spanish stuff).
Start work on related work section
Expand on the motivation section for spoken language translation, this might require a bit of research, but essentially why is this task useful to the world rather than to yourselves?
 Look at https://opus.nlpl.eu/ for possible parallel text sources for MT)  

### Asides

