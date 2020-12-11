# Import required libraries
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.firefox import GeckoDriverManager
import time
import csv
from bs4 import BeautifulSoup
import requests
import pathlib
import math
import os
import pandas as pd
import numpy as np
import re
from langdetect import detect

# Function to scrap the content desired
def scrap_book(html_path):
    # Get HTML
    page_soup = BeautifulSoup(open(html_path, encoding='utf-8'), features="lxml")

    # Title (to save as bookTitle)
    try:
        book_title = page_soup.find_all('h1')[0].contents[0].replace('\n', '').strip()
    except:
        book_title = ''
        
    # Series (to save as bookSeries)
    try:
        bookSeries = page_soup.find_all('h2')[2].contents[1].contents[0].replace('\n', '').strip()
    except:
        bookSeries = ''
    
    # Author(s), the first box in the picture below (to save as bookAuthors)
    try:
        auth_ini = page_soup.find_all('span', itemprop='name')
        bookAuthors = page_soup.find_all('span', itemprop='name')[0].contents[0].replace('\n', '').strip()
        for i in range(len(auth_ini) - 1):
            auth_aux = page_soup.find_all('span', itemprop='name')[i + 1].contents[0].replace('\n', '').strip()
            bookAuthors = bookAuthors + ', ' + auth_aux

    except:
        bookAuthors = ''
        
    # Ratings, average stars (to save as ratingValue)
    try:
        ratingValue = page_soup.find_all('span', itemprop='ratingValue')[0].contents[0].replace('\n', '').strip()
    except:
        ratingValue = ''
        
    try:
        ratings = page_soup.find_all('a', href="#other_reviews")
        ratingCount = -1
        reviewCount = -1
        for reviewCount in ratings:
            if reviewCount.find_all('meta', itemprop="ratingCount"):
                # Number of givent ratings (to save as ratingCount)
                ratingCount = reviewCount.text.replace('\n', '').strip().split(' ')[0].replace('\n', '').strip()
            elif reviewCount.find_all('meta', itemprop="reviewCount"):
                # Number of reviews (to save as reviewCount)
                reviewCount = reviewCount.text.replace('\n', '').strip().split(' ')[0].replace('\n', '').strip()
    except:
        ratingCount = ''
        reviewCount = ''

    # The entire plot (to save as Plot)
    
    try:
        plot_ini = page_soup.find_all('div', id="description")[0].find_all('span')[-1].contents
        Plot = str(page_soup.find_all('div', id="description")[0].find_all('span')[-1].contents[0]).replace('\n', '').strip().replace('"', " ").replace("'", " ").replace("\t", " ").replace("<br/>", " ").replace("<blockquote>", " ").replace("</blockquote>", " ").replace("<i>", " ").replace("</i>", " ").replace("<p>", " ").replace("</p>", " ").replace("<b>", " ").replace("</b>", " ").replace("<u>", " ").replace("</u>", " ").replace("<em>", " ").replace("</em>", " ").replace("<strong>", " ").replace("</strong>", " ").replace("<div>", " ").replace("</div>", " ").replace("#1", " ").replace("*", " ")
        if "ISBN" in Plot or "<a href=" in Plot:
            Plot = ''

        for i in range(len(plot_ini) - 1):
            plot_aux = str(page_soup.find_all('div', id="description")[0].find_all('span')[-1].contents[i + 1]).replace('\n', '').strip().replace('"', " ").replace("'", " ").replace("\t", " ").replace("<br/>", " ").replace("<blockquote>", " ").replace("</blockquote>", " ").replace("<i>", " ").replace("</i>", " ").replace("<p>", " ").replace("</p>", " ").replace("<b>", " ").replace("</b>", " ").replace("<u>", " ").replace("</u>", " ").replace("<em>", " ").replace("</em>", " ").replace("<strong>", " ").replace("</strong>", " ").replace("<div>", " ").replace("</div>", " ").replace("#1", " ").replace("*", " ")
            if "ISBN" in plot_aux or "<a href=" in plot_aux:
                plot_aux = ''
            Plot = Plot + ' ' + plot_aux.strip()
    except:
        Plot = ''
        
    # Number of pages (to save as NumberofPages)
    try:
        NumberofPages = page_soup.find_all('span', itemprop='numberOfPages')[0].contents[0].replace('\n', '').replace('pages', '').strip()
    except:
        NumberofPages = ''
        
    # Published (Publishing Date)
    try:
        PublishingDate = page_soup.findAll("div", {"class": "row"})[1].contents[0].replace('\n', '').replace('Published', '').split('by')[0].strip()
    except:
        PublishingDate = ''
        
    # Characters
    try:
        char_ini = page_soup.find_all("a", href=re.compile("/characters/"))
        Characters = page_soup.find_all("a", href=re.compile("/characters/"))[0].contents[0].replace('\n', '').strip()
        for i in range(len(char_ini) - 1):
            char_aux = page_soup.find_all("a", href=re.compile("/characters/"))[i + 1].contents[0].replace('\n', '').strip()
            Characters = Characters + ', ' + char_aux
    except:
        Characters = ''
        
    # Setting
    try:
        place_ini = page_soup.find_all("a", href=re.compile("/places/"))
        Setting = page_soup.find_all("a", href=re.compile("/places/"))[0].contents[0].replace('\n', '').strip()
        for i in range(len(place_ini) - 1):
            place_aux = page_soup.find_all("a", href=re.compile("/places/"))[i + 1].contents[0].replace('\n', '').strip()
            Setting = Setting + ' - ' + place_aux
    except:
        Setting = ''
    
    # Get Plot language
    try:
        language = detect(Plot)
    except:
        language = 'unk'
    
    # Return all the values retrieved as list
    result = [book_title, bookSeries, bookAuthors, ratingValue, ratingCount, reviewCount, Plot, NumberofPages, PublishingDate, Characters, Setting, language]
    return (result)



# Create function to change letters by positions in the alphabet
def alphabet_pos(string):
    # String: A string providing the letters to use
    # Get a dictionary to give to each letter a number according the position in the alphabet
    alphabet_dicc = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)} 
    # Convert string to lowercase
    string = string.lower()
    # Change each letter by its corresponding number
    string_as_num = [alphabet_dicc[letter] for letter in string if letter in alphabet_dicc]
    # Return string as number
    string_as_num = list(map(int,string_as_num))
    return string_as_num


# Define recursive function to calculate the maximum length of a subsequence of characters that is in alfabetical order
def max_len_sub_alpha_recursive(i, string_as_num):
    # i: position of the character where the string is included
    # string_as_num: list representing the numbers mapped from letters
    
    # Define the base case
    result = 1
    
    # Lopp over all the posibilites to the left of the position i of the string
    for j in range(i):
        # Check if the position j of the string is lower than the position i 
        # (for the sequence to be in alphabetical order)
        if string_as_num[j] < string_as_num[i]:
            # Calculate the result as the maximum of the actual result and the recursive function
            result = max(result, max_len_sub_alpha_recursive(j, string_as_num) + 1)
    return result

# Define the function that executes the max_len_sub_alpha function for each character of the string
def review_string(string):
    # Convert string to list of numbers
    string_as_number = alphabet_pos(string)
    
    # Get the length of string
    len_str = len(string)
    
    # Initialize the result
    result = 1
    
    # Loop over all the characters of the string
    for i in range(len_str):
        # Get the maximum result when appling the recursive function
        result = max(result, max_len_sub_alpha_recursive(i, string_as_number))
    
    return result

# Define dynamic programming function to calculate the maximum length of a subsequence of characters that is in alfabetical order
def max_len_sub_alpha_dyn_prog(string):
    # i: position of the character where the string is included
    # string_as_num: list representing the numbers mapped from letters

    # Convert string to list of numbers
    string_as_number = alphabet_pos(string)

    # Get the length of string
    len_str = len(string)

    # Preallocate auxiliar array of the size of string with the 
    # minimum length of subsequences (equals to 1 in each position)
    aux = [1 for i in range(len_str)]

    # Initialize the maximum length 
    result = 1

    # Loop over each letter starting from the second one
    for i in range(1, len_str):
        # Loop over all the letters that are before the ith letter
        for j in range(i):
            
            # Check if the actual letter (i) is higher than the jth letter
            if string_as_number[i] > string_as_number[j]:
                
                # Check if the actual length maximum subsequence is hogher than 
                # the stored in the (j+1)th position of the auxiliar array 
                if aux[j] + 1 > aux[i]:
                    
                    # Replace the actual length maximum subsequence by the one in the jth position adding 1
                    aux[i] = aux[j] + 1
                    
                    # Check if the actual length of maximum subsequence is higher than the previous one
                    if aux[i] > result:
                        # Find and store the length of maximum subsequence
                        result = aux[i]
    
    return result