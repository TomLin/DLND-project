## Unit Test
The unit tests provide some very insightful information on analyzing the correctness of the code you have written in terms of analyzing a few basic things like shape, kind of variable being used etc. Do have a good look at it as it would be useful in the future too.

## Build Network
### `get_inputs`
I would suggest you to look up sentdex's [video tutorials](https://www.pythonprogramming.net/machine-learning-tutorials/). Has some of the best tutorials for library usage for matplotlib, pandas and also for ML/DL related topics. He has explained the usage of a few basic tensorflow concepts too in the end of his ML tutorials

### `get_init_cell`
Very well done here. But try implementing the dropout for this. It might not make much of a difference for this particular use case but they are a very helpful method to overcome the over-fitting problem in neural networks. Do try to play around with the hyper parameters, which will help you with understanding the effects of these parameters on the model which will be of significant knowledge from my point of view.

Also good that you have increased the lstm_layers to 2 as it would make your network more sophisticated and a complex one which will be able to increase the level of the model being able to mimic human level interactions.

### `get_embed`
Great implementation here. I'd like to suggest an alternative approach that is just as useful as the one implemented here.

Suggestion: I'd like you to take a look at [tf.contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence) it can also be used for embedding.

### `build_nn`
Great work here. You have successfully applied embedding on the input data and also built a RNN.

Note, however, that tf.layers.dense would also work just as well, here (and it uses a linear activation by default). It's a little simpler/more straight-forward, but your choice works just as well in practice (unless you choose to include weights and biases initializers, which only the fully_connected version supports).

## Generate TV Script
### `pick_word`
Fantastic work on using the numpy's random.choice() function here. Most of the students tend to use the popular argmax function here tending the model to generate the same script all the time since it picks out the word with the maximum probability everytime the model is run.





