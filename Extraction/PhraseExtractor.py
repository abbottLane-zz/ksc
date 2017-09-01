import operator
import spacy
from spacy.symbols import *

class PhraseExtractor(object):
    def __init__(self):
        self.segmentor = spacy.load('en')
        self.phrase_labels= {nsubj, dobj, iobj, pobj, pcomp,xcomp, ccomp, attr} #amod, prep

    def extract_top_n_phrases_METHOD1(self, text, n=10):
        '''
        This method uses the NP chunker methods in spaCy to identify noun phrases in the sentences of a text
        :param text: The raw text (string) from which to extract phrases
        :param n: the number of most frequently-occurring phrases to return
        :return: a dictionary of the top n occurring phrases and their counts
        '''
        segmented_text = self.segmentor(text)
        phrases = list()
        for sent in segmented_text.sents:
            phrases.extend([s.orth_ for s in sent.noun_chunks])
        phrase_tuples = self._get_phrase_dict(phrases)
        top_n_phrases = self._pick_top_n_phrases(phrase_tuples)
        return top_n_phrases

    def extract_top_n_phrases_METHOD2(self, text, n=10):
        '''
        This method looks at the dependency parse produced by a text and uses arc labels to determine
         substrings that are phrase candidates
        :param text: The raw text (string) from which to extract phrases
        :param n: the number of most frequently-occurring phrases to return
        :return: a dictionary of the top n occurring phrases and their counts
        '''
        segmented_text = self.segmentor(text)
        phrases = list()
        for sent in segmented_text.sents:
            phrases.extend([s.text for s in self.iter_phrases(sent)])
        phrase_tuples = self._get_phrase_dict(phrases)
        top_n_phrases = self._pick_top_n_phrases(phrase_tuples)
        return top_n_phrases


    def _get_phrase_dict(self, phrases):
        '''
        Given a list of phrases (strings), count the occurences, return sorted dictionary of {phrase_str1:occurence_count, ...}
        :param phrases: a list of strings
        :return: descending-sorted dictionary of phrases and their counts
        '''
        p_dict=dict()
        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            if self._is_correct_length(phrase_lower):
                if phrase_lower in p_dict:
                    p_dict[phrase_lower] +=1
                else:
                    p_dict[phrase_lower]=1
        return sorted(p_dict.items(), key=operator.itemgetter(1), reverse=True)

    def _is_correct_length(self, phrase):
        '''
        Checks to make sure the NP chunk is between 2 and 10 tokens in length
        :param np: The span obj containing the NP tokens
        :return: True if the NP span fits the length restriction, otherwise False
        '''
        words = phrase.split()
        if len(words) > 1 and len(words) < 11:
            return True
        return False

    def iter_phrases(self, doc):
        '''
        Checks each word to see if it governs the phrase type we want. if it does, we yield the subtree which is
        the list of tokens which constitute the phrase.
        :param doc: a spaCy document object
        :return: a Phrase object wrapper of the governing word's subtree
        '''
        for word in doc:
            #print (word.orth_ + ": " + word.dep_ + ": " + str(word.dep) + ": " + word.pos_)
            if word.dep in self.phrase_labels:
                yield Phrase(word.subtree)



    def _pick_top_n_phrases(self, phrase_count_tuple, n=10):
        '''
        Assuming a SORTED list of ("phrase":count) tuples
        :param phrase_count_tuple: a SORTED list of ("phrase":count) tuples
        :return: a dictionary of the top n phrases and their counts, with no phrases being substrings of another
        '''
        final_n = list()
        for t in phrase_count_tuple:
            if len(final_n) ==0:
                final_n.append(t)
            else:
                substr_detected=False
                for i in range(len(final_n)):
                    if final_n[i][0] in t[0] or t[0] in final_n[i][0]:
                        substr_detected=True
                        break
                if not substr_detected:
                    final_n.append(t)
            if len(final_n) == n:
                return final_n
        return final_n


class Phrase(object):
    '''
    Initialized with a token generator that return all the tokens in a desired phrase
    '''
    def __init__(self, token_gen):
        self.token_generator = token_gen
        self.text = self._get_phrase_text(token_gen)

    def _get_phrase_text(self, tok_gen):
        tokens_text = [x.orth_ for x in tok_gen]
        text=" ".join(tokens_text)
        return text

