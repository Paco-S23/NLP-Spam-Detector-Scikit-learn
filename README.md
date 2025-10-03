Spam Detector with Logistic Regression and Scikit-learn
This project is an implementation of a Machine Learning model to classify SMS messages as spam or ham (not spam).

Project Objective
The main goal is to build a simple yet effective text classifier, demonstrating a complete Machine Learning workflow: from data loading and preparation to training and evaluating a predictive model.

Dataset
The "Email Spam Detection" dataset was sourced from Kaggle, published by user mfaisalqureshi. The original dataset is the "SMS Spam Collection" from the UCI Machine Learning Repository, which contains 5,572 English text messages, each labeled as either spam or ham.

Methodology
The process followed to build the model was as follows:

Data Loading: The Pandas library was used to load and explore the spam.csv file.

Text Preprocessing: The text messages were transformed into a numerical matrix using Scikit-learn's CountVectorizer. This method converts each message into a vector based on word frequency.

Data Splitting: The dataset was divided into three parts to ensure an objective evaluation of the model:

Training Set (70%): For the model to learn the patterns.

Validation Set (15%): To fine-tune and evaluate the model during development.

Test Set (15%): To measure the final performance of the model on unseen data.

Model Training: A Logistic Regression model was trained, which is ideal for binary classification problems like this one.

Evaluation: The model's performance was measured on the test set, using accuracy as the primary metric.

Results
The final model achieved an accuracy of 98.56% on the test set, demonstrating high effectiveness in differentiating between spam and non-spam messages.
