import re
'''
This is a library of regex patterns I was contemplating using to create features by collapsing topical terms. 
Some of these word groups are flawed in that they don't stick to a single semantic class, but rather attempt
to group all relevant words for a given topic all together. The thinking behind this approach is incredibly flawed, 
and doesn't work at all. Those groups are not being used in this system.

There are other groups (journalism, performing arts, crafts) that collapse words belonging to the same semantic class.
Using these dictionaries to normalize tokens actually results in a modest performance gain, so those methods are employed 
in the final model training/execution code.
'''
def sub_journalism_terms(text):
    regexes =  [" radio ", " magazine ", " documentary ", " newsletter ", " web magazine ",
                " travel blog ", " newsletters ", " interview ", " interviews ", " online publication ",
                " online publications ", " essays ", " essay ", " social media ", " photo journalism ",
                " public television ", " public broadcasting ", " television news ", " audio journalism ",
                " audio documentaries "]
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " MODE_OF_MEDIA_DELIVERY ", text)
    return text

def sub_performing_arts_terms(text):
    regexes = " performing arts ", " performance ", " dance ", " spoken word ", " opera "
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " PERFORMING_ARTS_TERM ", text)
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
    regexes = [" quilting ", " embroidery ", " crafts ", " mosaic ",
              "porcelain", " glassware ", " pottery ", " tatting ", " rug-making ", " stained glass ",  " wrought iron ",
              " glass making ", " glassblowing ", " glass etching ", " weaving ", " knitting ", " crochet ", " felting ",
              " floral design ", " bouquet ", " leather crafting ", " carving ", " wood carving ", " scrapbooking ",
               " calligraphy "," origami ", " decoupage ", " bookbinding ", " embossing ", " calligraphy ",
               " carpentry ", " upholstery ", "woodworking",
               " intarsia ", " stonemason ", " jewellery ", " silversmith ", " handicraft ", " charms "
    ]
    combinedRegex = re.compile('|'.join('(?:{0})'.format(x) for x in regexes))
    text = re.sub(combinedRegex, " CRAFT_TERM ", text)
    return text