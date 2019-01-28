[012819]
Following notes are only for personal study purpose.

## Transfer Learning

**Transfer Learning : instead of training from scratch, utilize pre-trained weights obtained from a large dataset.[1]**
Because the fine-tuned model doesn't have to learn from scratch, it can generally reach higher accuracy with much less data and computation time than models that don't use transfer learning.[2]

Possible methods : UMLFit, OpenAI Transformer, ELMo, BERT
<br>
Purpose of transfer learning : Make NLP classification easier.[2]
<br>
Simple(Shallow) Approach : Single layer of weights (embeddings) - word2vec, GloVe <br>
Pre-trained word embeddings have been immensely influential, but they have a major limitation: they only incorporate previous knowledge in the first layer of the model. The rest of the network still needs to be trained from scratch.[3]

=> Going from just initializing the first layer of models to pretraining the entire model with hierarchical representations.
If learning word vectors is like only learning edges, these approaches are like learning the full hierarchy of features, from edges to shapes to high-level semantic concepts.[3]
Pre-trained representations learn tasks related to low-level morphological and syntactic tasks at lower layers and longer range semantics at higher layers.[4] The pre-trained language models indeed capture similar properties as computer vision models pretrained on ImageNet.[5]

<br>
1. UMLFit :<br>
Deep Learning has seen some success in NLP. Most of successful NLP tasks share the characteristic of utilizing large amounts of labeled data, which limited the scope of research to those who were able to collect and label huge datasets.[2]


<br>
References :
[1] https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde
[2] http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html
[3] http://ruder.io/nlp-imagenet/
[4] http://aclweb.org/anthology/D18-1179
[5] http://ruder.io/10-exciting-ideas-of-2018-in-nlp/
