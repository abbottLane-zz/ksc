import re

def sub_journalism_terms(text):
    regexes = "newspaper", "writing", "book", "books", "story"
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, "JOURNALISM_TERM", text)
    return text

def sub_photography_terms(text):
    regexes = "portrait", "shot", "picture", "photograph", "photo"
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, "JOURNALISM_TERM", text)
    return text

def sub_food_terms(text):
    regexes = "recipes", "bake", "cook", "spice", "kitchen"
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, "JOURNALISM_TERM", text)
    return text
