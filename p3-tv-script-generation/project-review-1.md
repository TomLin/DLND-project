# Summary
You have a good start on this project. The hard part is all done (well...except for get_batches, that is). Please review my comments, incorporate the suggested changes, and I think you can relatively quickly complete this project with flying colors. The only required change is to fix your get_batches function to correct the last list of the batches data structure and pick_word to add some randomness to your word selections, preferably using the numpy random.choice function as it will produce a distribution of words matching the probability distribution provided to you. Since you need to make those updates anyway, I hope you will also take the time to adjust a few hyper-parameters (particularly rnn_size) to further improve your network loss metric and overall TV Script quality.

BTW, I really enjoyed reading your code. Your thoughts are clear, your code is easy to read, and this is one of the better submissions I've reviewed for this project. Keep up the great work!

Insight: :notebook: The same type of RNN network you are building in this project is used in many real-world applications. Reference, in particular, the recent announcement of Google Duplex: https://ai.googleblog.com/2018/05/duplex-ai-system-for-natural-conversation.html which does some amazing things using an RNN very similar to the one you just built!

## Unit Test
If you aren't already a believer in test-driven development, check out these 9 reasons to bring this methodology into your standard software development workflow: https://www.madetech.com/blog/9-benefits-of-test-driven-development

## Preprocessing: word to int
1. Creating these types of lookup tables is a very important step in any deep learning project that uses text as input. The network would have a difficult time working with the text directly, so we need to convert every word in the text input to a number that can be processed by the network. At the end of processing, we will need to convert the numbers (outputs after computation) back to text again.
2. Note that in this project, you will use an embedding that maps each unique word to a multi-dimensional vector (typically, 200-300 dimensions). You will provide a random initial embedding (get_embed) and the actual embedding vectors will be learned. So, the two lookup tables you have just created (from strings to ints and back to strings) are just the first (and last) step in mapping words to something the network can train against and use to generate the text for your version of a Simpsons TV script.
3. Creating tokens for punctuation is only one of many possible data pre-processing steps you might consider when working with RNNs. This blog post (https://datascience.stackexchange.com/questions/11402/preprocessing-text-before-use-rnn) discusses a few others in the context of sentiment analysis (a common RNN task). Sentiment analysis is a different use-case from the text-generation example of this project, but these pre-processing tasks are applicable to many types of text-based deep learning projects:
    - Converting text to lowercase to avoid training the network to distinguish between upper and lower case versions of words.
    - Removing Numbers if they are not relevant to a task (such as sentiment analysis) they may be removed to let the network concentrate on other, more important distinctions.
    - Removing Punctuation – In this project, you included special code to handle punctuation , as it is important in generating human-like output. However, in other use cases, punctuation may not add value and can be removed.
    - Removing Stop Words – Stop words are words that are so common (like “the” “an”, etc. in English) that they can often be ignored for some use cases (like sentiment analysis, for example).
    - Stripping White Space so the network doesn’t “waste time” trying to differentiate “ “ from “ “, etc..
    - Stemming transforms word variations to a root form. This can include removing common endings (such as “s”, “ed”, etc.) and even performing even more extreme abbreviations.
    - Lemmatisation is similar to stemming, it transforms each word to its dictionary base form.
    - Removing Sparse Terms if we are not interested in infrequently-used terms, they can be removed.

## Build Network
### `get_inputs`
1. TF Placeholders are used to reserve a place in the TensorFlow computation graph for the input values that you will feed into the network.
2. In contrast, TF variables are used to hold values which will be updated in a TensorFlow session – such as the trainable values (biases and weights) in your RNN.

### `get_init_cell`
Be aware that the functions you are using abstract away all the details required to make an effective RNN network. You should understand what goes on under the hood. In addition to the Udacity videos (which are very good) I highly recommend reading a blog post by Christopher Olah. IMO, it is the very best article on LSTMs around, and well worth a read: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

### `get_embed`
Word2Vec isn’t used in this project, but it is an embedding method you should be familiar with and seriously consider for any project that requires a network to understand word relationships. It's a very powerful unsupervised technique to convert a dictionary of words to embedded vectors. It groups words in the embedding space in a way that lexical relationships can be recreated from standard vector math on the embedding vectors. A famous example: King – Man + Woman = Queen. This blog post by Chris McCormick is an excellent explanation of how Word2Vec and the embedding process in general works: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model <br/>
Note that Chris’ blog post is in two parts. The first part covers basics of the skip-gram model, while the second half goes into more detail on Word2Vec embedding. This technique doesn’t apply directly to this project (where the provided code learns an embedding specific to this project’s vocabulary), but you should be familiar with Word2Vec for any future real-world projects where your network will be working with text and grammar.

### `build_rnn`
TensorFlow provides several different methods for constructing RNNs. You can read about them here: https://www.tensorflow.org/api_guides/python/nn#Recurrent_Neural_Networks

In particular, three key RNN components include:

- tf.nn.dynamic_rnn (the one you used) makes use of a TensorFlow tf.While loop to dynamically construct the execution graph at run time. Surprisingly, the graph creation is faster than the static RNN implementation without a runtime performance penalty.
- tf.nn.bidirectional_dynamic_rnn - Builds independent forward and backward RNNs.
- tf.nn.raw_rnn - This function is a more primitive version of dynamic_rnn, but provides more direct access to the inputs each iteration. It also provides more control over when to start and finish reading the sequence, and what to emit for the output.

### `build_nn`
After all the emphasis on non-linear activation functions in this course, you might wonder why you have been asked to use a linear activation function in this project. Neural networks often use non-linear activation functions (such as sigmoid or RELU) in hidden layers but may use a linear activation function in the final output layer. A linear activation in the final layer scales the outputs continuously across the full range of values and is the best way to compare relative values of the various outputs. In this script-writing RNN, you are not trying to identify a single-best word to select (a RELU or other non-linear activation function would be good for this purpose). Rather, you want the output to represent the relative probabilities of each potential word to be selected – so a linear activation is appropriate here.

## Training
1. Note that getting the loss below 1.0 is a key objective requirement, but not the only measure of success on this project. You should also ensure that loss has plateaued at a low value before stopping training. Also, while num_epochs and learning_rate are key to getting the necessary loss value, the other factors affect the quality of the generated script and are just as important as the loss value in determining overall success on this project.
- num_epochs: Your choice of num_epochs is good enough to meet the requirements of this project since the loss does decrease below the required value of 1.0. In a "real-world" project, however, you would want to continue training until the loss plateaus and shows no signs of improvement. Your loss appears to still be dropping at the end of training, indicating that your network would continue to improve if you gave it more time to work.
- batch_size : is smaller than it probably needs to be and could potentially be increased 4x-8x depending on the size of the available GPU memory. This primarily affects training time, not the quality of the final results. Remember to tune this to the largest value that your GPU will support for the most efficient training. GPUs need to be kept fed with the biggest chunks of data they can handle!
    - Insight: If you were running on your CPU (with no GPU), the batch value wouldn’t have a big impact either way. The bottlenecks that occur between the main processor and GPU are what benefit from larger batch sizes. That bottleneck is not really a factor when no GPU is present and all processing is performed on the CPU.
    - Pro Tip: Batch sizes should always be a power of two as TensorFlow will optimize your computations when that is the case (your choice of 128 does meet this recommended practice).
    - Pro Tip: Bottom line: raise batch_size (in powers of 2) until your GPU complains it's out of memory, and then back it down to a happy value.
- rnn_size: This is an acceptable value, and has enough cells to produce good results on this task, particularly since you included multiple layers of LSTM cells in your architecture. I have noticed, however, that networks with 2x-3x this number of LSTM cells produce noticeably better output (TV scripts) for this project.
- embed-dim: Typically, embed dims are about 200-300 dimensions, so your choice is good. Note that the corpus in this project is very small compared to the large vocabularies you will see referenced in the embedding literature. So, something on the low end of this range might be preferable for this project, and from that point of view your choice is perfect!
- seq_length: The rubric states that seq_length should match the structure of the data, which I admit is a little vague. You have selected a value that is just a bit larger than the number of words in four lines of text (reference the statistics reported in the first executable cell of this project). There is some debate on what the best value for this parameter should be. I’m of the opinion that in order to let the network maintain some additional context between individual sentences, 3-4 lines of text is appropriate for this parameter. So, I like your choice!
- learning_rate: Looks very reasonable. This is something that you need to trade off with num_epochs, and your choices work well together here (as evidenced by the reported loss as you trained the network).
    - Pro Tip: If loss jumps up and down too much, that is an indication that learning_rate needs to be decreased. If loss decreases smoothly and monotonically, you're learning rate is good (or, could potentially be increased if training is taking too long).
- show_every_n_batches: You want to know both the value of the loss at the end of training, but also how it behaves during training -- and your choice achieves this goal.
    - Pro Tip: Ideally, you would want to print every epoch (as you have done). In real-world projects, you would generally plot the loss and more data will produce smoother plots.

## Generate TV Script
### `pick_word`
Unfortunately, this is a function I'm going to request that you update. Your current implementation always picks the single most-probable word from the available choices and probabilities. However, you should add a little variety/randomness to your selection. In other words, don't simply pick the single-most-probable word for each selection, but rather try to return words matching the specified probabilities. For example, if the probabilities vector indicates there is a 60% chance of returning "x" and 40% chance of returning "y", don't just return the most probable "x" every time, but rather, return "x" 60% of the time and "y" 40% of the time). Here's a hint on something that might help (you can accomplish the desired result in one line of code using this numpy function): https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html


