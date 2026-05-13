# ASR_project
Automated Speech Recognition (ASR) project built from scratch using python and PyTorch.
Our training data was given by Gdańsk University of Technology. However, it is not perfect.
It contains:
  20 hours of speech, about 800.000 phonemes, 24 different authors
The dataset was created using Montreal Forced Aligner (https://montreal-forced-aligner.readthedocs.io/en/latest/)
It turns audio + text into a labeled intervals of particular phonemes in a .TextGrid file

* I also created my own little dataset made for fine tuning, it contains about 15 minutes of my voice. I also like to thank my friend - a poet called "Brutus" for giving us access to his poems. We used them for our model.

The dataset is too small for a transformer, the model we trained is a CRNN (convolutional + recurrent neural network)

We trained 2 models for changing 
  * time frame window model
  * CTC (Connectionist Temporal Classification) loss function model

In both models we augment and regularize our data heavily.
 * we create more data adding a random noise and changing the amplitude of the voice randomly
 * we mask our spectrogram windows in both time and frequency domain

By these augmentations our model (with the same architecture!) went from 70% accuracy on a validation set into a 74,7% on a validation set!

And it works, it is not perfect and a silent room is recommended, but it predicts the phonemes that occured.

We still need another model (a language one) for making a better guess and corrections for what words were actually spoken.


 
