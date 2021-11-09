# Flower classification using cosine similarity - Ajay K R

import cv2
import math
import os

directories = os.listdir("./flowers/flower_photos/train")


# Read all images from all categories
def get_knowledge_base():
    knowledge_base = []
    for i in range(len(directories)):
        category = directories[i]
    
        flower_data = []
        images = os.listdir("./flowers/flower_photos/train/" + category + "/")
    
        for img in images:
            im = cv2.imread("./flowers/flower_photos/train/" + category + "/" + img, cv2.IMREAD_GRAYSCALE)
            flower_data.extend([im])
        knowledge_base.extend([flower_data])    
    return knowledge_base
    #[category][img][row][col]


# Load knowledge base
knowledge_base = get_knowledge_base()

def classify(img):
    classification = []
    for each_category in knowledge_base:
        category_score = 0                  # Assume score for a particular category is 0 initially
        
        # Compute highest score for each category (of images) agaist input image
        for each_image in each_category:
            score = similarity_score(img, each_image) # Compute similarity score for the input image against an image in a category
            if(score > category_score):               # If the score is higher, then that is the score for that category
                category_score = score 
            
        category_score = category_score/len(each_category)
        classification.extend([category_score])  
    return classification


# Compute similarity between two images
def similarity_score(input_image, knowledge_base_image):
    row = len(input_image)
    col = len(input_image[0])
    if len(input_image) > len(knowledge_base_image):
        row = len(knowledge_base_image)
    
    similarity = 0
    for i in range(row):
        similarity = similarity + cos_similarity(input_image[i], knowledge_base_image[i])
    
    return similarity


# Compute the dot product between two vectors
# If the vectors don't have the same dimension, then vector with smaller dimension gets filled with 0 in missing places
# Missing places are considered 0
def dot_product(vector_A, vector_B):
    col = len(vector_A)
    if len(vector_A) > len(vector_B):
        col = len(vector_B)             # Ignore the rest of the values as they are considered to be 0                          
    
    dot_product = 0
    for i in range(col):
        dot_product = dot_product + (vector_A[i]*vector_B[i])
    
    return dot_product


# Compute the absolute value of a vector
def length_of_vector(vector):
    length = dot_product(vector, vector)
    length = math.sqrt(length)
    return length


# Compute cosine similarity between two vectors
def cos_similarity(vector_A, vector_B):
    cos_theta = (dot_product(vector_A, vector_B))/(length_of_vector(vector_A) * length_of_vector(vector_B))
    return cos_theta



# Driver code
test_image = cv2.imread("./flowers/flower_photos/test/daisy/134409839_71069a95d1_m.jpg", cv2.IMREAD_GRAYSCALE)
classification = classify(test_image)
classification






