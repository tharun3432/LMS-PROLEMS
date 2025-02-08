#!/usr/bin/env python
# coding: utf-8

# # 1

# In[ ]:


# Declare variables of different data types
integer_var = 10         # Integer
decimal_var = 3.14       # Float
string_var = "Hello, Jupyter!"  # String
boolean_var = True       # Boolean

# Print each variable and its type
print("Integer Variable:")
print("Value:", integer_var)
print("Type:", type(integer_var))

print("\nFloat Variable:")
print("Value:", decimal_var)
print("Type:", type(decimal_var))

print("\nString Variable:")
print("Value:", string_var)
print("Type:", type(string_var))

print("\nBoolean Variable:")
print("Value:", boolean_var)
print("Type:", type(boolean_var))


# # 2

# In[2]:


# Creating a List
my_list = [10, 20, 30, 40, 50]

# Accessing elements based on index
first_element = my_list[0]  # Accessing the 1st element
third_element = my_list[2]  # Accessing the 3rd element
last_element = my_list[-1]  # Accessing the last element

print("List:", my_list)
print("First Element:", first_element)
print("Third Element:", third_element)
print("Last Element:", last_element)


# In[3]:


# Creating a Tuple
my_tuple = ('apple', 'banana', 'cherry', 'date', 'elderberry')

# Accessing elements based on index
second_element = my_tuple[1]  # Accessing the 2nd element
fourth_element = my_tuple[3]  # Accessing the 4th element
last_element = my_tuple[-1]  # Accessing the last element

print("Tuple:", my_tuple)
print("Second Element:", second_element)
print("Fourth Element:", fourth_element)
print("Last Element:", last_element)


# In[4]:


# Creating a Dictionary
my_dict = {'name': 'Alice', 'age': 25, 'city': 'New York', 'job': 'Engineer', 'salary': 75000}

# Accessing elements based on keys
name = my_dict['name']  # Accessing value associated with 'name'
age = my_dict['age']  # Accessing value associated with 'age'
city = my_dict['city']  # Accessing value associated with 'city'

print("Dictionary:", my_dict)
print("Name:", name)
print("Age:", age)
print("City:", city)


# # 3

# In[15]:


# Function to calculate grade based on average
def calculate_grade():
    # Taking input for marks in three subjects
    subject1 = float(input("Enter marks for Subject 1: "))
    subject2 = float(input("Enter marks for Subject 2: "))
    subject3 = float(input("Enter marks for Subject 3: "))

    # Calculating the average
    average = (subject1 + subject2 + subject3) / 3

    # Determining the grade
    if average >= 90:
        print("Grade: A")
    elif 80 <= average < 90:
        print("Grade: B")
    elif 70 <= average < 80:
        print("Grade: C")
    else:
        print("Grade: Fail")

# Call the function
calculate_grade()


# # 4

# In[17]:


# Function to calculate the sum of all even numbers between 1 and n
def sum_of_evens(n):
    # Initialize sum
    total = 0

    # Loop through numbers from 1 to n
    for i in range(1, n + 1):
        if i % 2 == 0:  # Check if the number is even
            total += i

    return total

# Input: Positive integer n
n = int(input("Enter a positive integer n: "))

# Check if the input is valid
if n > 0:
    # Call the function and display the result
    result = sum_of_evens(n)
    print(f"The sum of all even numbers between 1 and {n} is: {result}")
else:
    print("Please enter a positive integer.")


# # 5

# In[19]:


# Function to calculate the frequency of each word
def word_frequency(text):
    # Convert text to lowercase and split into words
    words = text.lower().split()

    # Create a dictionary to store word frequencies
    frequency = {}

    # Count the occurrences of each word
    for word in words:
        word = word.strip(",.?!;:'\"")  # Remove punctuation
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1

    return frequency

# Input: Text from the user
text = input("Enter a text: ")

# Call the function and get the word frequencies
frequencies = word_frequency(text)

# Display the word frequencies
print("\nWord frequencies:")
for word, count in frequencies.items():
    print(f"{word}: {count}")


# In[1]:


pip install nltk spacy


# In[ ]:


import nltk
nltk.download()


# In[ ]:


import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text using NLTK and spaCy
def preprocess_text(text):
    # Convert the text to lowercase
    text_lower = text.lower()

    # Tokenize the text using NLTK
    tokens = word_tokenize(text_lower)

    # Remove stopwords using NLTK
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # Use spaCy for lemmatization
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    return lemmatized_tokens

# Sample text input
text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction 
between computers and humans through natural language. NLP allows computers to understand, interpret, 
and generate human language in a way that is meaningful.
"""

# Preprocess the text
processed_text = preprocess_text(text)

# Display the processed text
print("Processed Text:")
print(" ".join(processed_text))


# # 6

# In[1]:


import re

def extract_emails(text):
    # Regular expression pattern for extracting email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)

# Test the function
test_string = 'Contact us at support@example.com and sales@example.org.'
emails = extract_emails(test_string)
print(emails)


# # 7

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
texts = ["Spam messages are annoying", "I won a lottery", "This is a normal message", "Click here to claim your prize", "Meeting at 10 AM"]
labels = [1, 1, 0, 1, 0]  # 1 for spam, 0 for normal

# Convert text data into numerical format
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Accuracy:", accuracy)


# # 8

# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample dataset
texts = ["Buy now and win a prize", "Normal email content", "Congratulations, you won!", "This is an important update", "Win a free gift card now"]
labels = [1, 0, 1, 0, 1]  # 1: Spam, 0: Not Spam

# Convert text data to numerical format
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

y = np.array(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# In[ ]:




