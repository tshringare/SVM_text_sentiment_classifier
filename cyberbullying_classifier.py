import joblib
from model_builder import pre_classification

# this file loads the model made by the model builder file, and compares it with the best model if included

# the below try and except block is to check if the model is built or not, if not built, it will direct the user
# to run the model builder first
try:
    with open('model.joblib', 'rb') as f:
        m_v = joblib.load(f)
        saved_model = m_v[0]
        vector = m_v[1]
except FileNotFoundError:
    print("the model has not been built, please run the model_builder.py file first.")
    exit()

try:                                                # try and except block to see if the best model is saved or not
    with open('best_model.joblib', 'rb') as f:      # if it's not, then will run with the normal model itself
        b_m_v = joblib.load(f)
        best_model = m_v[0]
        bVector = m_v[1]
except FileNotFoundError:
    print("best model is not saved, executing with the built model")
    tweet = r"I hate you"     # change this variable to try out different tweets

    print("tweet to be classified", tweet, "\n Built Model (f=220) classification:\n",
          saved_model.predict(pre_classification(tweet, vector))[0])

tweet = r"The extreme antibody reaction from those who fear free speech says it all"
# change this variable to try out different tweets

print("tweet to be classified", tweet, "\n Built Model (f=220) classification:\n",
      saved_model.predict(pre_classification(tweet, vector))[0],
      "\nBest Model (f=20000) classification:\n",
      best_model.predict(pre_classification(tweet, bVector))[0]
      )
