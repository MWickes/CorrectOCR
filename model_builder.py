import collections
import json
import os


def load_interview(filename, header=0):
    with open(filename, 'rb') as f:
        data = [i.decode('utf-8') for i in f]

    return data[header:]


# Load the files of confusion counts, remove any keys which are not single
# characters, remove specified characters, and combine into a single
# dictionary.
def load_confusion_counts(file_list, remove=[]):
    # Outer keys are the correct characters. Inner keys are the counts of
    # what each character was read as.
    confusion = collections.defaultdict(lambda: collections.Counter())
    for filename in file_list:
        with open(os.path.join('train/HMMtrain/',filename), 'rb') as f:
            counts = json.load(f, encoding='utf-8')
            for i in counts:
                confusion[i].update(counts[i])

    # Strip out any outer keys that aren't a single character
    confusion = {key:value for key, value in confusion.iteritems()
                 if len(key) == 1}

    for unwanted in remove:
        if unwanted in confusion:
            del confusion[unwanted]        

    # Strip out any inner keys that aren't a single character.
    # Later, these may be useful, for now, remove them.
    for outer in confusion:
        wrongsize = [key for key in confusion[outer] if len(key) != 1]
        for key in wrongsize:
            del confusion[outer][key]

        for unwanted in remove:
            if unwanted in confusion[outer]:
                del confusion[outer][unwanted]

    return confusion

# Get the character counts of the training files. Used for filling in 
# gaps in the confusion probabilities.
def interview_char_counts(file_list, remove=[], header=0):
    char_count = collections.Counter()
    for filename in file_list:
        interview = load_interview(os.path.join('train/parallelSource/',filename), header)
        c = collections.Counter(''.join(interview))
        char_count.update(c)

    for unwanted in remove:
        if unwanted in char_count:
            del char_count[unwanted]

    return char_count


# Create the emission probabilities using confusion counts and character
# counts. Optionally a file of expected characters can be used to add
# expected characters as model states whose emission probabilities are set to
# only output themselves.
def emission_probabilities(confusion, char_counts, alpha,
                           remove=[], char_file=None):
    # Add missing dictionary elements.
    # Missing outer terms are ones which were always read correctly.    
    for char in char_counts:
        if char not in confusion:
            confusion[char] = {char:char_counts[char]}
            
    # Inner terms are just added with 0 probability.
    charset = set().union(*[confusion[i].keys() for i in confusion])
            
    for char in confusion:
        for missing in charset:
            if missing not in confusion[char]:
                confusion[char][missing] = 0.0
    
    # Smooth and convert to probabilities.
    for i in confusion:
        denom = sum(confusion[i].values()) + (alpha * len(confusion[i]))
        for j in confusion[i]:
            confusion[i][j] = (confusion[i][j] + alpha) / denom

    # Add characters that are expected to occur in the interviews.
    if char_file is not None:
        with open(char_file, 'rb') as f:
            extra_chars = set(json.load(f, encoding='utf-8'))
        # Get the characters which aren't already present.
        extra_chars = extra_chars.difference(set(confusion))
        extra_chars = extra_chars.difference(set(remove))

        # Add them as new states.                
        for char in extra_chars:
            confusion[char] = {i:0 for i in charset}
        # Add them with 0 probability to every state.
        for i in confusion:
            for char in extra_chars:
                confusion[i][char] = 0.0
        # Set them to emit themselves
        for char in extra_chars:
            confusion[char][char] = 1.0

    return confusion
    
    
# Create the initial and transition probabilities from the corrected
# interviews in the training data.
def init_tran_probabilities(interview_list, alpha,
                            remove=[], header=0, char_file=None):
    tran = collections.defaultdict(lambda: collections.defaultdict(int))
    init = collections.defaultdict(int)
    
    for filename in interview_list:
        interview = load_interview(os.path.join('train/parallelSource/',filename), header)

        for line in interview:
            for word in line.split():
                if len(word) > 0:
                    init[word[0]] += 1
                    # Record each occurrence of character pair ij in tran[i][j]
                    for i in xrange(len(word)-1):
                        tran[word[i]][word[i+1]] += 1

    # Create a set of all the characters that have been seen.
    charset = set(tran.keys())
    charset.update(set(init.keys()))
    for key in tran:
        charset.update(set(tran[key].keys()))

    # Add characters that are expected to occur in the interviews.
    if char_file is not None:
        with open(char_file, 'rb') as f:
            extra_chars = json.load(f, encoding='utf-8')
        charset.update(set(extra_chars))

    for unwanted in remove:
        if unwanted in charset:
            charset.remove(unwanted)
        if unwanted in init:
            del init[unwanted]
        if unwanted in tran:
            del tran[unwanted]
        for i in tran:
            if unwanted in tran[i]:
                del tran[i][unwanted]

    # Add missing characters to the parameter dictionaries and apply smoothing.
    init_denom = sum(init.values()) + (alpha * len(charset))
    for i in charset:
        init[i] = (init[i] + alpha) / init_denom
        tran_denom = sum(tran[i].values()) + (alpha * len(charset))
        for j in charset:
            tran[i][j] = (tran[i][j] + alpha) / tran_denom

    # Change the parameter dictionaries into normal dictionaries.
    init = {i:init[i] for i in init}
    tran = {i:{j:tran[i][j] for j in tran[i]} for i in tran}

    return init, tran


def parameter_check(init, tran, emis):
    all_match = True
    if set(init) != set(tran):
        all_match = False
        print 'Initial keys do not match transition keys.'
    if set(init) != set(emis):
        all_match = False
        print 'Initial keys do not match emission keys.'
    for key in tran:
        if set(tran[key]) != set(tran):
            all_match = False
            print 'Outer transition keys do not match inner keys: {}'.format(key)
    if all_match == True:
        print 'Parameters match.'
    return all_match



num_header_lines = 12
smoothing_parameter = 0.0001
remove_chars = [' ', '\n', '\r', u'\ufeff']

# Select the gold files which correspond to the confusion count files.
gold_files = []
confusion_files = []
for filename in os.listdir('train/HMMtrain/'):
    confusion_files.append(filename)
    # [:-10] is to remove '_confusion'
    gold_files.append('c_' + os.path.splitext(filename)[0][:-10] + '.txt')

confusion = load_confusion_counts(confusion_files, remove_chars)
char_counts = interview_char_counts(gold_files, remove_chars, num_header_lines)

# Create the emission probabilities from the confusion counts and the character counts
emis = emission_probabilities(confusion, char_counts, smoothing_parameter, remove_chars, 
                              char_file='resources/additional_characters.txt')

# Create the initial and transition probabilities from the gold files
init, tran = init_tran_probabilities(gold_files, smoothing_parameter,
                                     remove_chars, num_header_lines, 
                                     char_file='resources/additional_characters.txt')

if parameter_check(init, tran, emis) == True:
    with open('resources/hmm_parameters.txt','wb') as f:
        json.dump((init, tran, emis), f)
