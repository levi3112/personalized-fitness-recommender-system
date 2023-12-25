import cv2
import pytesseract
import matplotlib.pyplot as plt
import re
import nltk

tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    sentences = nltk.sent_tokenize(text.lower()) # Tokenize the text into sentences
    number_regex = r'\b\d{2,3}\b' # Define the regular expression for a number
    final_numbers = {} # number : priority
    for sentence in sentences: # Iterate over the sentences
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        for i in range(len(words)):
            # Check if the word is a 2 or 3 digit number
            if re.match(number_regex, words[i]):
                final_numbers[words[i]] = 1 # gives a priority as 1

    # Iterate over the keys of the final_numbers dictionary
    for number in final_numbers.keys():
        # Find all occurrences of the number in the text
        occurrences = re.findall(r'\b' + number + r'\b', text)
        # The count of the number is the length of the occurrences list
        count = len(occurrences)
        # Print the number and its count
        print(f"The number {number} appears {count} times in the text.")
        # set the priority of the number, based on the count
        p = final_numbers[number] / count
        final_numbers[number] = p

    