# Cross-Cohort Music Recommendations

## Project Goal:
This project uses Machine Learning to group songs together based on their musical "DNA" (like tempo, energy, and lyrics) rather than just their genre. 

The goal is to create a **Recommendation System** that can suggest a Jazz song to a Hip-Hop fan because the two tracks share similar underlying patterns.



## How it works:
1. **The Data:** I used a dataset of songs that had already been simplified using PCA (a way to boil down complex data into its most important parts).
2. **Finding the Groups:** I used an algorithm called **K-Means Clustering**. 
3. **Picking the "K":** To figure out how many groups of music to create, I used the **"Elbow Method."** It looks for the point where adding more groups doesn't actually help the model learn more.
    * *My finding:* 6 groups seemed to be the "sweet spot."
4. **The Results:** The model successfully sorted songs into 6 distinct Clusters. For example, it identified a specific group (Cluster 4) that contains Hip-Hop tracks like *Shiki No Uta* and *Wake Up*.



## What's inside this repo?:
* `clustering.ipynb`: The main "brain" of the project where the analysis happens.
* `kmeans_model.pkl`: The finished, saved version of the model that can be used in other apps.
* `music_pca_data.csv`: The music data used for training.

## Challenges & Observations:
While the model did a great job separating genres like Jazz, Soul, and Pop, the separation was almost *too* perfect. In my notes, I mentioned that I need to double-check if there was an error in the data processing (like accidentally leaving the genre labels in the training data). 

**Next Step:** Experiment with different data scaling to make the clusters even more accurate.