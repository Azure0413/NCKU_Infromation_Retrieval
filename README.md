# NCKU_Infromation_Retrieval
To understand and be able to apply information retrieval technology in automated biomedical literature search. Students will participate in intensive computer programming projects and will enhance their skills via research to various search technologies. This course also includes paper presentation and final project as well. Students will be expected to complete all course requirements upon their participation.

## HW1
Implement a full-text retrieval tool (i.e. search engine) for a set of text documents. Specifically, your system will be able to perform document retrieval according to specified keyword(s) and then display in an easily visualization way. The tool is able to calculate the document statistics (such as number of characters, number of words, number of sentences(EOS), etc.) and to determine how many sentences in documents using smart method (for example, rule-based approach). Computer languages are not restrictive.

## HW2
First of all, implement the Zipf Distribution computation (or frequency spectrum for terms) for both a set of text documents from PubMed. The size of text document sets could range from 10 to 10000, depends on your intention. You have to preprocess the text index (token) set from document collection. Secondly, implement the Porter ’s algorithm as a functional module on your own software platform for the same set of text documents. Then, compare the difference.  
In this project, basically your system or software platform will be able to compute text distribution from a set of documents. Match (or partial matching) process can be done using the dynamic programming-based Edit distance computation. Your retrieval results can be displayed in a format of indicating the location(s) and/or partial matching of the query keywords in each document, etc. Computer languages are not limited.

## HW3
Implement the Word Embedding Technique(word2vec) for a set of text documents from PubMed with same subject. The size of text document sets could range from 1000 to 10000, depends on your original intention. You have to preprocess the text set from document collection. 
In this project, you can choose one of the 2 basic computational models: 
1. Continuous Bag of Word (CBOW): use a window of word to predict the middle word
2. Skip-gram (SG): use a word to predict the surrounding ones in window.  Window size is not limited. Computer languages are not limited.

## HW4
In this project, you are asking to implement and analyze “term weighting” technology for text documents in the vector space before executing the Porter’s algorithm. At least 2~3 types of TF-IDF and/or modified TF-IDF methods1, such as sentences or paragraphs are considered in this project. Then, you need to rank the most important key sentences in the paragraphs of documents based on term weighting and similarity measure, in which you have to choose one reasonable ranking and similarity computation method. 

## Final Project
Focused on using food and cooking images, which are valuable information sources to predict recipe 
● For textual information, the text data is obtained from sentence transformer model, such as BERT or RoBERTa(with Pretraining) 
● To extract image features, U-Net or Contrastive Language-Image Pre-training (CLIP) can be used 
● The individual retrieval results are based on the cosine similarity of vector representations 
● To use both of the features, the individual retrieval results are combined using average or maximum scores, etc. 
![image](https://github.com/Azure0413/NCKU_Infromation_Retrieval/blob/main/info.png)
