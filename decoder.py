import os
import csv
import json
import cStringIO
import codecs
import re
import collections
import itertools



class Decoder(object):

    def __init__(self, hmm, word_dict, prev_decodings=None, substitutions=None):
        self.hmm = hmm
        self.word_dict = word_dict
        if prev_decodings is None:
            self.prev_decodings = dict()
        else:
            self.prev_decodings = prev_decodings


    def decode_word(self, word, k):
        if len(word) == 0:
            return [''] + ['',0.0] * k

        if word in self.prev_decodings:
            return [word] + self.prev_decodings[word]

        k_best = self.hmm.k_best_beam(word, k)        
        k_best = [element for subsequence in k_best for element in subsequence]
        self.prev_decodings[word] = k_best

        return [word] + k_best


    def multichar_variants(self, word, original, replacements):
        variants = [original] + replacements
        variant_words = set()
        pieces = re.split(original, word)
        
        # Reassemble the word using original or replacements
        for x in itertools.product(variants, repeat=word.count(original)):
            variant_words.add(''.join([elem for pair in itertools.izip_longest(
                pieces, x, fillvalue='') for elem in pair]))
            
        return variant_words


    def strip_punctuation(self, word):
        punctuation = re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        word = re.sub('[' + punctuation + ']+', '', word)    
        return word
    


class HMM(object):

    def __init__(self, initial, transition, emission):
        self.init = initial
        self.tran = transition
        self.emis = emission
        
        self.states = initial.keys()
        self.symbols = emission[self.states[0]].keys()

    def viterbi(self, char_seq):
        # delta[t][j] is probability of max probability path to state j
        # at time t given the observation sequence up to time t.
        delta = [None] * len(char_seq)
        back_pointers = [None] * len(char_seq)

        delta[0] = {i:self.init[i] * self.emis[i][char_seq[0]]
                    for i in self.states}

        for t in xrange(1, len(char_seq)):
            # (preceding state with max probability, value of max probability)           
            d = {j:max({i:delta[t-1][i] * self.tran[i][j]
                        for i in self.states}.iteritems(),
                       key=lambda x: x[1]) for j in self.states}
            
            delta[t] = {i:d[i][1] * self.emis[i][char_seq[t]]
                        for i in self.states}
            
            back_pointers[t] = {i:d[i][0] for i in self.states}

        best_state = max(delta[-1], key=lambda x: delta[-1][x])

        selected_states = [best_state] * len(char_seq)
        for t in xrange(len(char_seq) - 1, 0, -1):
            best_state = back_pointers[t][best_state]
            selected_states[t-1] = best_state

        return ''.join(selected_states)

    # Beam search, not Viterbi.
    def k_best_beam(self, word, k):
        # Single symbol input is just initial * emission.
        if len(word) == 1:
            paths = [(i, self.init[i] * self.emis[i][word[0]])
                     for i in self.states]
            paths = sorted(paths, key=lambda x: x[1], reverse=True)
        else:
            # Create the N*N sequences for the first two characters
            # of the word.
            paths = [((i, j), (self.init[i] * self.emis[i][word[0]]
                               * self.tran[i][j] * self.emis[j][word[1]]))
                     for i in self.states for j in self.states]

            # Keep the k best sequences.
            paths = sorted(paths, key=lambda x: x[1], reverse=True)[:k]

            # Continue through the input word, only keeping k sequences at
            # each time step.
            for t in xrange(2, len(word)):
                temp = [(x[0] + (j,),
                         (x[1] * self.tran[x[0][-1]][j] * self.emis[j][word[t]]))
                         for j in self.states for x in paths]
                paths = sorted(temp, key=lambda x: x[1], reverse=True)[:k]

        
        return [(''.join(seq), prob) for seq, prob in paths[:k]]



class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    Source: https://docs.python.org/2/library/csv.html
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        #self.writer.writerow([s.encode("utf-8") for s in row])
        
        # Modification to avoid attempting to encode non-unicode entries.
        self.writer.writerow([s.encode("utf-8") if type(s) == unicode else s
                              for s in row])
        
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
