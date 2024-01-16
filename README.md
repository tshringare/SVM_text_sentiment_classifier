<h1>Cyberbullying Classifier</h1>
<h2>Introduction</h2>
<p>This project uses the Cyberbullying classification dataset to train a machine learning model to classify whether some text/post is an instance of cyberbullying. The model should be more accurate if the given text is a tweet
as the model is trained on a dataset containing tweets (this is an assumption). The machine learning model used is SVM (Support vector machines) as it is able to handle a large number of features relatively well, allowing it to
perform better in text classification problems. The repository contains the dataset, the model builder, which would allow the user to build their own models by fine tuning a parameters, a general model (f=220), a "best" model (f=20000) and 
a cyberbullying text classifier which classifies the given input according to the built model and the "best" model. </p>

<h2>Dataset Used</h2>
<p>The Dataset used is the Cyberbullying Classfication by Larxel, found on Kaggle:https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification/data</p>
<p>Said Dataset is protected under Attribution 4.0 International (CC BY 4.0) license, details for which can be found here: https://creativecommons.org/licenses/by/4.0/</p>

<h2>Libraries Used</h2>
<ul><li>SKLearn</li>
<li>Pandas</li>
<li>NLTK</li>
<li>Joblib</li></ul>

<h2>Working</h2>
<p>The model builder uses the SKlearn library for model building, the pandas library, and the NLTK library for any preprocessing.The Pre-processing done involves the standard text classification procedures, such
as the removal of stop words, stemming, and lematizing, along with several other procedures, such as convertion of web urls, email addresses, hashtags etc. to a standard notation. After these steps, the tfidf vectorizer is
applied to the dataset, which converts each record into a vector with values being assigned to each word depending on their importance in the record (tweet) with relation to their frequency in the entire dataset. According
to these values, a specific number of features are selected (default = 220).</p>
<p>The Model was then trained using the SVC model builder from SKLearn. The kernel(s) used during the creation of the model are linear.</p>
<p>The data set is split into a training and testing dataset, with 30% of the data being used for testing and the rest being used for training</p>
<p>For Classification of a specific tweet/text using the model, the string is first put through the same pre-processing steps as the dataset, then fit into the tfidf vector format used in the dataset. The resulting vector is then
used for classification.</p>
<p>The model and the tfidf vector is then stored in a .joblib file for use. The model can be tried out using the cyberbullying classifier python file.</p>

<h2>How to use</h2>
<p>Run the model_builder.py file. One may make changes to the parameters inside the .py file as required. Importantly, the number of features used while training can be changed, which is found in the vector variable inside
the process fucntion. Note that changing this affects the accuracy of the resulting model and larger values (>100) are suggested for text classification. Another thing to note is that for very large number of features, the
stemming and lemmatizing results in a lower accuracy model than if the text is left as is. Finally, after the model is built, classification of an example tweet is shown, along with the calculated accuracy of the model itself.
One may also choose to change the train test split, found under main</p>

<p>Next, run the cyberbullying_classifier.py to try inputing a specific text and seeing its classification according to their newly built model and the "best" model (f=20000). The best model is named as such due the increase
in features resulting in a reduction of accuracy.</p>

<p>If one wishes to use the models themselves, the .joblib files for the models are stored in the same directory and are also found on github. The generated model is simply named model.joblib while the "best" model is named
best_model.joblib. These models can be loaded using joblib library and can be used for classification after fitting the text to be classified in the specific tfidf vector. One can see how to do this by referring to the cyberbullying_classifier.py</p>

<p>One may also choose to use this model_builder to build models around other text classification datasets by replacing the .csv file in the model_builder.py, <b>as long as they follow the same format as the used dataset</b>, that is, the datset
should contain one column of text and another column of it's classification.</p>

