# Recommendation Engine
 
 A book recommendation Engine that uses Clustering and User Based Collaborative Filtering.
 The dataset was obtained [here](https://www.kaggle.com/bahramjannesarr/goodreads-book-datasets-10m), and the GoodReads website was further scraped to obtain the Genres of the books.
 
 The file userProfile.py contains the code for the model, where we implemented a Clustered User based Collaborative Filtering. The file itemProfile.py is our attempt to do something similar with item based Collaborative filtering. The final run can be observed in the notebook Recommendation. 
 
 The final preprocessed data is in finaldset.csv - there was a LOT of work to reach this point, cleaning the text data, removing a lot of duplicates, handling NANs, binning the reviews, etc.
 
 Overall, we are satisfied with the results produced from the model (User-based), as this was a good learning experince to understand the inner workings of one type of recommender systems, with additional functionality(clustering) added to it.
 
