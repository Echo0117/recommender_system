# Recommender_System

# Instructions

## Enviroment
we used
```
python version：3.8
```

## Dependencies
you can install all the dependencies by
```
pip install -r requirements.txt
```

There may occure some errors with `scikit-surprise` and `faiss`, but it is only used in the comparision part for svd optimisation.
It won't affect for the use of the main program. So it is okay that you do NOT install it if you only want to play with the basic 
implementation of out recommender system :)

With pip:

`pip install numpy`

`pip install scikit-surprise`

With conda:

`conda install -c conda-forge scikit-surprise`

`conda install -c pytorch faiss-cpu`

## Usage
you can just run
```
python run.py
```

It will lanuch a web page, you can input the `user_id`

![img_3.png](images/img_3.png)

and there are two recommender methods you can choose
`content_based` or `collacorative_filtering`
![img_1.png](images/img_1.png)

choose the the number of k for top k movies

![img_2.png](images/img_2.png)

Then you will get the result.


![img_4.png](images/img_4.png)

You can continue to play with it by input `yes`

![img_5.png](images/img_5.png)

# Code structure
``` 
├── data_processing         //data processing fuctions
│   ├── __init__.py
│   ├── embeddings.py
│   └── preprocessing.py
├── dataset                //dataset that we used including the pre-saved matrix
│   ├── ml-latest-small
│   └── saved_embeddings
│       ├── movies_tfidf_embeddings.pkl
│       └── use_rating_matrix_embeddings.pkl
├── evaluations           // The evaluations scripts we used
│   ├── __init__.py
│   ├── p_top_k_evaluation_script.py
│   └── rmse_evaluation_script.py
├── recommender_system    // main functions of this project are here
│   ├── __init__.py
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   ├── evaluation.py
│   └── optimization      //The optimization methods we used
│       ├── __init__.py
│       ├── dimensionality_reduction.py
│       ├── faiss_retrieval.py
│       └── lsh_retrieval.py

├── test                    //Unit tests for the functions
│   ├── __init__.py
│   ├── test_collaborative_filtering.py
│   ├── test_content_based.py
│   ├── test_data_processing.py
│   └── test_evaluation.py
└── web                    //A simple web demo for play with the results
    ├── __init__.py
    └── recommender_web.py
├── requirements.txt      
├── run.py                 //main function to run this program
├── ext.py                 //ext tools to initilize some variables
├── config.json            //config files
├── config.py
├── data_analysis.ipynb    //some analysis of the input data
```

# Credits

[Yihan Zhong](https://github.com/YIHAN-ZHONG)