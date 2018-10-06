import csv
import json

import decoder




def load_hmm(filename):
    with open(filename, 'rb') as f:
        h = decoder.HMM(*json.load(f, 'utf-8'))
    return h


def load_dictionary(filename):
    with open(filename, 'rb') as f:
        worddict = set([i.decode('utf-8').strip() for i in f])
    return worddict


def load_text(filename, header=0):
    with open(filename, 'rb') as f:
        data = [i.decode('utf-8') for i in f][header:]
    words = []
    temp = []
    for line in data:
        for char in line:
            # Keep newline and carriage return, but discard other whitespace
            if char.isspace():
                if char == '\n' or char == '\r':
                    if len(temp) > 0:
                        words.append(''.join(temp))
                    words.append(char)
                    temp = []
                else:
                    if len(temp) > 0:
                        words.append(''.join(temp))
                    temp = []
            else:
                temp.append(char)
                
    # Add the last word
    if len(temp) > 0:
        words.append(''.join(temp))
                
    return words


def load_csv_unicode(filename, delimiter='\t', quoting=csv.QUOTE_NONE):
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=quoting)
        data = [[unicode(element, 'utf-8') for element in line]
                for line in reader]
    return data




num_header_lines = 12
k = 4
decoded_header = ['Original']
for i in xrange(k):
    decoded_header.extend(['{}-best'.format(i+1),
                           '{}-best prob.'.format(i+1)])

# Load previously done decodings if any
##prev_decodings = dict()
##
##for filename in os.listdir('decoded/'):
##    for line in load_csv_unicode('decoded/'+filename)[1:]:
##        prev_decodings[line[0]] = line[1:]
##
### Load the rest of the parameters and create the decoder
##dec = decoder.Decoder(load_hmm('hmm_parameters.txt'),
##                      load_dictionary('resources/SAMPLE_dictionary.txt'),
##                      prev_decodings)
##
### Decode files
##for filename in os.listdir('to_decode/'):
##    words = load_text('to_decode/'+filename, num_header_lines)
##    decoded_words = []
##    
##    for word in words:
##        decoded_words.append(dec.decode_word(word, k))
##
##    with open('decoded/'+filename, 'wb') as f:
##        writer = dec.UnicodeWriter(f,
##                                   dialect=csv.excel_tab,
##                                   quoting=csv.QUOTE_NONE,
##                                   quotechar=None)
##        writer.writerows(decoded_words)


