# Relation Networks

Implementation of the bAbi task from [A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427) in PyTorch using TorchText.

`run.sh` does everything, i.e. getting the data and running the model.

The model will run until the `test_loss` has not improved for 3 epochs.

The paper is pretty light on the details for the bAbi task. However, it does state:

- uses the 20 previous sentences
- each sentence fed through a 32 unit LSTM, which is shared across all sentences
- the question is fed through a 32 unit LSTM, separate from the sentence LSTM
- `g` is a four layer MLP with 256 units per layer
- `f` is a three layer MLP with 256, 512 and 159 units per layer
- optimizer is Adam with a learning rate of 2e-4

Here's what I've done to fill in some blanks:

- instead of using the 20 previous sentences, I've used all supporting sentences per question (of which there are up to 8)
- if a question has less than 8 supporting sentences, the remainder are filled with a `<pad>` token
- `g` is a concatenation of (object, question, object_index), where object index is the "position of the supporting sentence", i.e. the first supporting sentence has an index of 0, the second has an index of 1, etc.
- the vocabulary for the questions and supporting sentences are shared
- the question and supporting sentences are passed through an embedding layer before being fed to the LSTM
- this embedding layer converts the words to 32 dimensional vectors
- this embedding layer is shared between the questions and supporting sentences
  - note: I tried with the questions and supporting sentences have their own distinct vocabularies and embedding layers and it gave worse results on the whole
- dropout was used after the embedding layer and between each MLP layer
  - values of dropout >0.5 seem to hinder performance greatly
  - even a small amount of dropout makes the model very unstable (final accuracies vary wildly for same values of dropout)
- ReLU was used between each MLP layer
- Xavier initialization was used
- the number of units in the final MLP layer of `f` is 59 and not 159, this is because the vocabulary size of the answer is 59 (this may have been a typo in the original paper)

Other notes:

- training seems very unstable, accuracy seems to vary wildly between 40% to 80%
- tried having each object concatenated with every other object as well as the question, this gave more stable results however wasn't able to achieve near the best accuracy (best for single object was ~88%, best for two objects was ~82%)

TODO:

- tweak the hyper-parameters I've set to get the same results of the paper, current best is ~88%
