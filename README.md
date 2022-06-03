# cookies-cognition
Cookies and Cognition Analytics for Austerweil Lab

clean_x.ipynb shows basic cleaning of the three data sets

transcription_variations.ipynb examines differences between the transcription results of shared participants

preprocess_merge_x.ipynb modifies the data so that receipts are sufficiently similar for the merge algorithm to be feasible

merge.py contains the algorithm that aligns and merges grocery data via a word vector language model implementing Word Mover's Distance. 

merge.ipynb executes the merge algorithm on our data sets and analyzes the results