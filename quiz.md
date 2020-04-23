# Word2Vec Embeddings Quiz

## General Word2Vec Concept

This section will assess your understanding of general Word2Vec and Training Optimization concept.

1. The following three statements are about Word2Vec, choose the **CORRECT** statement(s): (a) Word2Vec is an unsupervised learning, even though we use a neural network to train the data. (b) In Skip-Gram architecture, the values that are projected onto the hidden layer is the word vector for a certain word in vocabulary. (c) In CBOW architecture, the values that are projected onto the hidden layer is the word vector for a certain word in vocabulary.
  - [ ] (a) and (b)
  - [ ] (a) and (c)
  - [ ] (b) and (c)
  - [ ] all of the above
  - [ ] none of the above

2. Regardless of the Word2Vec architecture, how many training examples will be generated from the sentence **"roses are red violets are blue"** if the window size is 3?
  - [ ] 18
  - [ ] 21
  - [ ] 24
  - [ ] 27
   
3. Suppose we have a vocabulary of 5,000 words and learning 100-dimensional Word2Vec embeddings. There will be 500,000 weights of the output matrix to be updated during backpropagation if we don't use a training optimization. Which one of the following statements is **CORRECT**?
  - [ ] There are also 500,000 weights of the input matrix to be updated during backpropagation.
  - [ ] Using `negative = 20` or K = 20 means that only 20 weights (instead of 500,000) of the output matrix will be updated on each batch.
  - [ ] Using `ns_exponent = 0` or p = 0 means that every single word has a probability of 0.0002 to be choosen as negative sample.
  - [ ] Using hierarchical softmax, the size of our output matrix is still 100 x 5,000.
  
## Training Word2Vec

This section will assess your understanding of training Word2Vec using `gensim` package.

4. Choose one parameter that will speed up the training process if we increase its value, assuming the other parameters remain unchanged.
  - [ ] min_count
  - [ ] negative
  - [ ] size
  - [ ] window
  
5. The following statements are all correct in the process of building a Word2Vec model using `gensim`, **EXCEPT** ...
  - [ ] Stemming is rarely performed because a word can lose its semantic meaning.
  - [ ] The training process is separated into three steps for clarity and monitoring.
  - [ ] We doesn't have to set up anything for printing out a training report.
  - [ ] The training sentence is in the form of a "list of lists of tokens".

## Case Study: Recommender System

In this section, we will build a system to recommend similar products based on online transaction history which provided in `Online Retail.xlsx` inside the folder `dataset`. This dataset contains 541909 transactions and 8 columns as follow:

* `InvoiceNo`: Invoice number. a unique number assigned to each transaction
* `StockCode`: Product/item code. a unique number assigned to each distinct product
* `Description`: Product name
* `Quantity`: The quantities of each product per transaction
* `InvoiceDate`: Invoice Date and time. The day and time when each transaction was generated
* `UnitPrice`: Price of each unit product
* `CustomerID`: Customer number. a unique number assigned to each customer
* `Country`: The country where each transaction was generated

After loading the dataset, make sure you have performed the following preprocessing step:
* Drop all the rows with missing value
* Convert each columns to its proper data type
* Remove leading and trailing whitespace on column `Description`
* Prepare a `code_to_name` and/or `name_to_code` dictionary mapping, assuming the product name used is the latest name by date

In order to get the same result, pre-trained model `recommender.model` is provided to you inside the folder `models`. The model has a vocabulary of 3196 unique products and size of the word vector is 100 dimensions. The parameters used to train the model are as follow:
* size = 100
* window = 10
* min_count = 5
* sg = 1
* hs = 0
* negative = 15
* alpha = 0.005
* seed = 123
* workers = 1
* epochs = 10

Please make sure to include the `Callback` class before loading the pre-trained model:

```
from gensim.models.callbacks import CallbackAny2Vec

class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss
```

6. Let's say we build a recommender system so that items that have at least a similarity score of **0.90** with previous transactions will appear on the "Product Recommendation" section of a website. Suppose there is a customer who only purchased one item, namely **"BLUE PAISLEY TISSUE BOX"**. How many **new products** will appear on their "Product Recommendation" section?
  - [ ] 9
  - [ ] 10
  - [ ] 11
  - [ ] 12

If we use the system stated on question number 6, the number of recommended products will be different for each customer. Suppose we change how our recommender system works. Now, for each customer, we want the model to recommend exactly **10 most similar products** based on entire purchase history of a user. 

First, you may need to find a list of purchased products for each customer:

```
purchased_product_for_each_customer = retail.groupby("...")["..."].apply(list)
```

7. Now, let's analyze customer with CustomerID "13160". Is there any product that they have purchased before, but the model recommends again? If yes, what is the product description?
  - [ ] 15CM CHRISTMAS GLASS BALL 20 LIGHTS
  - [ ] BLUE SPOT CERAMIC DRAWER KNOB
  - [ ] DRAWER KNOB CERAMIC IVORY
  - [ ] HEART WREATH DECORATION WITH BELL
  - [ ] None of the top 10 recommended products have been purchased by the CustomerID "13160"
