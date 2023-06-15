# VisTranformer
Visual Transformers
<center><img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png" alt="Alternative text"/></center> 
<center><figcaption>Fig 1. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale."https://arxiv.org/pdf/2010.11929.pdf. </figcaption></center>                 


In the context of sequence-to-sequence modelling applications like natural language processing (NLP). Their superior performance to LSTM-based Recurrant neural network gained them a powerful reputation, thanks to their ability to model long sequences. A couple of years ago, transformers have been adapted to the [visual domain](https://arxiv.org/abs/2010.11929) and suprisingly demonstrated better performance compared to the long standing convolutional neural networks conditioned to large-scale datasets. Thanks to their ability to capture global semantic relationships in an image, unlike, CNNs which capture local information within the vicinty of the convolutional kernel window.


## 1. Image Patches and Linear Mapping

### A) Image Patches
Transfomers were initially created to process sequential data. In case of images, a sequence can be created through extracting patches. To do so, a crop window should be used with a defined window height and width. The dimension of data is originally in the format of *(B,C,H,W)*, when transorfmed into patches and then flattened we get *(B, PxP, (HxC/P)x(WxC/P))*, where *B* is the batch size and *PxP* is total number of patches in an image. 


B) Linear Mapping
Afterwards, the input are mapped using a linear layer to an output with dimension d i.e. (B, PxP, (HxC/P)x(WxC/P)) → (B, PxP, d). The variable d can be freely chosen.



## 2. Insert Classifier Token and Positional embeddings

### A) Classifier Token

Beside the image patches, also known as tokens, an additional special token is appended to the the input to capture desired information about other tokens to learn the task at hand. Lateron, this token will be used as input to the classifier to determine the class of the input image. To add the token to the input is equivilant to concatentating a learnable parameter with a vector of the same dimension *d* to the image tokens. 

### B) Positional Embedding

To preserve the context of an image, positional embeddings are associated with each image patch. Positional embeddings encodes the patch positions using sinusoidal waves, however, there are other techniques. We follow the definition of positional encoding in the original transformer paper of [Vaswani et. al](https://arxiv.org/abs/1706.03762), which sinusoidal waves. You'll be required to implement a function that creates embeddings for each coordinate of every image patch. 



## 3. Encoder Block

## 3. Encoder Block



This is the challenging part as it will be required from you to implement the main elements of an encoder block. A single block contains layer normalization (LN), multi-head self-attention (MHSA), and a residual connection.  

### A) Layer Normalization
[Layer normailzation](https://arxiv.org/abs/1607.06450), similar to other techniques, normalizes an input across the layer dimension by subtracting mean and dividing by standard deviation. You can instantiate layer normalization which has a dimension *d* using [PyTorch built-in function](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
### B) MHSA
  
  
 The attention module derives an attention value by measuring similarity between one patch and the other patches. To this end, an image patch with dimension *d* is linearly mapped to three vectors; query **q**, key **k**, and value **v** , hence a distint linear layer should be instantiated to get each of the three vectors. To quantify attention for a single patch, first, the dot product is computed between its **q** and all of the **k** vectors and divide by the square root of the vector dimension i.e. *d* = 8. The result is passed through a softmax layer to get *attention features* and finally multiple with **v** vectors associated with each of the **k** vectors and sum up to get the result. This allows to get an attention vector for each patch by measuring its similarity with other patches.
 
 
 It is highly recommended to define a seperate class for MHSA as it contains several operations.

### C) Residual Connection

Residual connections (also know as skip connections) add the original input to the processed output by a network layer e.g. encoder. They have proven to be useful in deep neural networks as they mitigate problems like exploding / vanishing gradients. In transformer, the residual connection is adding the original input to the output from LN &rarr; MHSA. All of the previous operations could be implemented inside a seperate encoder class.

The last part of an encoder, is to a inser another residual connection between the input to the encoder and the output from the encoder passed through another layer of LN &rarr; MLP. The MLP consists of 2 layers with hidden size 4 times larger than *d*.


## 4. Classification Head
The final part of implemeting a transformer is adding a classification head to the model inside *LightViT* class. You can simply use a linear classifier i.e. a linear layer that accepts input of dimension *d* and outputs logits with dimension set to the number of classes for the classification problem at hand.

## 5. Model Train
At this point you have completed the major challenge. Now all you need to do is to implement a standard script for training and testing the model. 

