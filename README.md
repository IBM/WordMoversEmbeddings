# WordMoversEmbeddings
WordMoversEmbeddings(WME) is a simple code for generating univeral text embedding of variable length, including sentence, parapgrah, and document.

This code is a simple implementation (mix of Matlab, Matlab MEX, and C) of the WME in (Wu et al, "Word Mover’s Embedding: From Word2Vec to Document Embedding", EMNLP'18). We refer more information about WME to the following paper link: https://arxiv.org/abs/1811.01713 and the IBM Research AI Blog: https://www.ibm.com/blogs/research/2018/11/word-movers-embedding/. 
 

# Prerequisites

The preprocessing code needs python 2.7 with several packages. If you use Anaconda, you can set up vitual environment: 

conda create -n yourenvname python=2.7 anaconda

You will also need to download one of most popular pre-trained embeddings such as Word2Vec or GloVe. You can download them from these links:

For Word2Vec: https://drive.google.com/open?id=1RfimQWYm3C3KQNUQJ0EnUdz0xqWsoP8Q <br/>
For GloVe: https://drive.google.com/open?id=1Ul__vKCUXFANr4M79_QU1HpPgvSLlqlx <br/>
For PSL: https://drive.google.com/open?id=1Kgoksc27NffaDO9-uGzuEASL4ybLpYgu <br/>


# How To Run The Codes
Note that, to get the good performance for your applications, the hyperparameters DMax, gamma, and even lambda_inverse (for text classification using SVM) have to be searched (using cross validation or other techniques) in order to find the best number of these hyperparameters. This is a crucial step for supervised machine learning tasks.  

To generate the WME for your NLP applications, you need:

(1) Prepare raw text data to its corresonding Matlab data format:

    a) prepare original text format as follows:

    doc1_label_ID \t word1 word2 word3 word4
    doc2_label_ID \t word1 word2 word3 word4
    ..

    Note: see data_proc/twitter.txt as an example.

    b) generate the Matlat format of original text as follows:

    python get_word_vectors.py twitter.txt o twitter.mat

    Note: set up the right path for reading pre-trained word embeddings. 

    c) run postproc_generateDataset.m to generate final Matlab dataset with five different train/test split (if the data has no split). 

(2) If the emd mex file (in both linux and mac) throws an error, you need to run build_emd.m in the utilities folder. 

(3) If you would like to use same datasets in our EMNLP'18 paper, please download the datasets here:

    For text classification tasks: https://drive.google.com/open?id=175Mj8-2Vj0uWxVyUHFoGlbrITQ2CFDdW 
    For textual similarity tasks: https://drive.google.com/open?id=1TtszuLFt0iwD-zqMjvbLgCJiwQZQ6bXa

(4) Open Matlab terminal console and run wme_gridsearch_CV.m for performing K-fold cross validation for searching good hyperparameters 
    The WME embeddings that performs the best on the dev data will be saved. 

(5) Test the model by running the following code wme_VaryingR_allSplits_CV_R256.m using best parameters from CV
    The testing result on different data splits will be saved. 

(6) To generate WME embedding only, please run this code wme_Genfea_example.m Note that there are no default numbers for the hyperparameters DMax, gamma. You should searching for the best numbers before generating text embeddings for your applications. In general, the larger the parameter R is, the better quality of embedding is. 



# How To Cite The Codes
Please cite our work if you like or are using our codes for your projects! Let me know if you have any questions: lwu at email.wm.edu. 

Lingfei Wu, Ian E.H. Yen, Kun Xu, Fangli Xu, Avinash Balakrishnan, Pin-Yu Chen, Pradeep Ravikumar, and Michael J. Witbrock, "Word Mover's Embedding: From Word2Vec to Document Embedding", EMNLP'18. 

@InProceedings{wu2018word, <br/> 
  title={Word Mover's Embedding: From Word2Vec to Document Embedding}, <br/> 
  author={Wu, Lingfei and Yen, Ian EH and Xu, Kun and Xu, Fangli and Balakrishnan, Avinash and Chen, Pin-Yu and Ravikumar, Pradeep and Witbrock, Michael J}, <br/> 
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing}, <br/>
  year={2018} <br/> 
}


------------------------------------------------------
Contributors: Lingfei Wu <br/>
Created date: November 28, 2018 <br/>
Last update: November 28, 2018 <br/>

