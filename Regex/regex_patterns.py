import re
'''
This is a library of regex patterns I was contemplating using to create features by collapsing topical terms. Theoretically, the logic behind this attempt at dimension reduction was flawed, and empirically, 
this hurt classification performance, so none of this code is being used.
'''
def sub_journalism_terms(text):
    regexes = " media ", " magazine ", " writing ", " journalist ", " journalism "
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " JOURNALISM_TERM ", text)
    return text

def sub_photography_terms(text):
    regexes = " photography ", " images ", " camera ", " photograph ", " photo "
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " PHOTOGRAPHY_TERM ", text)
    return text

def sub_food_terms(text):
    regexes = " recipes ", " coffee ", " cook ", " beer ", " kitchen "
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " FOOD_TERM ", text)
    return text

def sub_art_terms(text):
    regexes =  " artist ", " artwork ", " painting "
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " ART_TERM ", text)
    return text

def sub_design_terms(text):
    regexes = " design ", " designs ", " designer ", " designed "
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " DESIGN_TERM ", text)
    return text

def sub_fashion_terms(text):
    regexes = " leather ", " clothing ", " skirts ", " fit ", " wear "
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " FASHION_TERM ", text)
    return text

def sub_craft_terms(text):
    regexes = " kiln ", " craft ", " crafty ", " glass "
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " CRAFT_TERM ", text)
    return text