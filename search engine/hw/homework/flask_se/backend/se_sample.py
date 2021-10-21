import nltk as n
import json
import math
import collections
'''
retrieve all abstarcts of papers as a list
return papers and list of abstracts without punctuation marks
'''
def get_abstract_list():
    # get all data from "paper.json"
    with open('./paper.json','r') as f:
        papers = json.load(f)
    
    # define the stop_words, original abstract list and initailize filted abstracts
    abstract_list = [item['abstract'] for item in papers]
    filtered_abstracts = []
    
    # filter every abstract in abstract list
    for abstract in abstract_list:
        # perform tokenization
        word_tokens = tokenizer.tokenize(abstract)
        # filter the stop word
        filtered_sentence = [w.lower() for w in word_tokens]
        filtered_abstracts.append(filtered_sentence) 
    
    return papers, filtered_abstracts

'''
stem all abstarcts in the list and create inverted index for it if it is not stop word
return inverted index before removing stop words
'''
def stem_and_index(abstracts_list):
    # initialize the stemmer and stemmed list
    invert_index = dict()
    # stem every abstract and create inverted index
    for i, abstract in enumerate(abstracts_list):
        # create inverted index
        for j, word in enumerate(abstract):
            stem_word = ps.stem(word)
            # if it is stop word, don't create index
            if stem_word not in stop_words:
                dic = dict()
                # add it to inverted index (word -> document which contains it)
                if stem_word in invert_index:
                    dic = invert_index[stem_word]
                else:
                    invert_index[stem_word] = dic
                # if word is the first time in this document, create new list
                if i not in dic:
                    dic[i] = []
                dic[i].append(j)
    
    return invert_index

''' 
return number of the phase contained by the document
document_id is a the number of document
'''
def count_phase_number(document_id, word_tokens):
    word = word_tokens[0]
    count = 0
    if word in invert_index and document_id in invert_index[word]:
        for index in invert_index[word][document_id]:
            i = index
            flag = True
            for w in word_tokens:
                if i >= len(abstracts_list[document_id]) or not w == ps.stem(abstracts_list[document_id][i]):
                    flag = False
                    break
                i += 1
            count += flag
    return count

'''
compute the cosine similarity between document weight and query weight
return a similarity number like 10
'''
def compute_weight_and_similarity(document_id, word_weight, phase_weight):
    # initialize document weight and number of document N
    document_weight = []
    N = len(abstracts_list)
    # dict to record if it be contained by the document weight
    unique_word = dict()
    # comput norm of the query_weight
    query_norm = 0
    query_weight = collections.OrderedDict()
    query_weight.update(word_weight)
    query_weight.update(phase_weight)
    for weight in query_weight.values():
        query_norm += (weight ** 2)
    query_norm = math.sqrt(query_norm)
    top5 = dict()
    min_word = ''

    # compute the one document weight of the word  
    for word in word_weight:
        weight = 0
        if word in invert_index:
            unique_word[word] = 1
            document_index = invert_index[word]
            if document_id in document_index:
                tf = len(document_index[document_id])
                idf =  math.log2(1.0*N/len(document_index))
                weight = tf * idf
                if len(top5) < 5:
                    if len(top5) == 0 or weight <= top5[min_word]:
                        min_word = word
                    top5[word] = weight
                elif weight > top5[min_word]:
                    top5.pop(min_word)
                    top5[word] = weight
                    min_word = word
        # add weight to the document weight list
        document_weight.append(weight)

    # compute the one document weight of the phase  
    for phase in phase_weight:
        word_tokenss = tokenizer.tokenize(phase)
        word_tokens = []
        for word in word_tokenss:
            stem = ps.stem(word)
            if not stem in stop_words:
                # avoid to add word in the phase to the document weight
                unique_word[stem] = 0
                word_tokens.append(stem)
        
        weight = 0
        if word_tokens[0] in invert_index:
            document_index = invert_index[word_tokens[0]]
            if document_id in document_index:
                # compute the tf
                tf = count_phase_number(document_id, word_tokens)
                # if tf > 0, then go to calculate idf
                if tf:
                    # compute the idf
                    document_count = 0
                    for id in document_index:
                        if count_phase_number(id, word_tokens):
                            document_count += 1
                    idf =  math.log2(1.0*N/document_count)
                    weight = tf * idf
                # add weight to the document weight list
        document_weight.append(weight)

    # compute all term weight in the document 
    for w in abstracts_list[document_id]:
        stem = ps.stem(w)
        if stem not in stop_words and stem in invert_index:
            if stem not in unique_word:
                tf = len(invert_index[stem][document_id])
                idf =  math.log2(1.0*N/len(invert_index[stem]))
                weight = tf * idf
                document_weight.append(weight)
                if len(top5) < 5:
                    if len(top5) == 0 or weight <= top5[min_word]:
                        min_word = stem
                    top5[stem] = weight
                elif weight > top5[min_word]:
                    top5.pop(min_word)
                    top5[stem] = weight
                    min_word = stem
                unique_word[stem] = 1
            else:
                unique_word[stem] += 1

    # compute similarity between document and query
    cosine_similarity = 0
    document_norm = 0
    for i, weight in enumerate(query_weight.values()):
        cosine_similarity += document_weight[i] * weight
    for weight in document_weight:
        document_norm += (weight ** 2)
    document_norm = math.sqrt(document_norm)

    unique_count = 0
    for unique in unique_word.values():
        unique_count += (unique == 1)
    # compute the cosine similarity
    if document_norm == 0:
        return {'document_id':document_id, 'similarity':0, 'document_vector_norm':0, 'unique_word_number': unique_count, 'top5_word_weight':top5}
    else:
        return {'document_id':document_id, 'similarity':cosine_similarity / document_norm / query_norm, 'document_vector_norm':document_norm, 'unique_word_number': unique_count, 'top5_word_weight':top5}

'''
handle the query, return top 5 id
1. stem the query and get query weight
2. compute document weight of the query
3. compute the similairity and sort
'''
def search(word_weight, phase_weight):
    # find documents which contain any of query item
    # compute similarity score of the article (similarity of Q and each D) and sort
    similarity = dict()
    for word in word_weight:
        # if word don't occur, document similarity don't need to be computed
        if word in invert_index:
            for id in invert_index[word]:
                # compute the similarity bettween document and query weight if haven't computed
                if id not in similarity:
                    similarity[id] = compute_weight_and_similarity(id, word_weight, phase_weight)
    for phase in phase_weight:
        word_tokens = tokenizer.tokenize(phase)
        word_tokens = [ps.stem(word) for word in word_tokens if not word in stop_words]
        # for phase, we only need to concentrate on the first word. if the first word has been invert index
        if word_tokens[0] in invert_index:
            for id in invert_index[word_tokens[0]]:
                # compute the similarity bettween document and query weight if haven't computed
                if id not in similarity:
                    similarity[id] = compute_weight_and_similarity(id, word_weight, phase_weight)
    
    sored_items = sorted(similarity.items(), key = lambda item:item[1]['similarity'], reverse=True)

    top5 = []
    for i in range(5):
        top5.append(sored_items[i][0])
    return top5

'''
input a query like 'a apple "english teacher"'
output a list of words and phases with stopwords removal
'''
def query_to_list(query):
    # split the query into list
    # for mark ' " ' Left double quotation mark
    flag = False
    # record phase (Wrapped in double quotes)
    s = ""
    # other character
    other = ""
    phase_tokens = []
    for ch in query:
        if ch == '"':
            if flag:
                phase_tokens.append(s)
                s = ""
            flag = not flag
        elif flag:
            s +=ch
        else:
            other += ch
    # merge other words and phase into a list, filter stopwords
    word_tokens = tokenizer.tokenize(other)
    return [w for w in word_tokens if not w in stop_words], phase_tokens

'''
input the filtered word, phase (tokenized, lowered and punctuation removed)
return word weight and phase weight
'''
def create_query_weight(filered_word, filtered_phase):
    # create query weight dictionary
    word_weight = collections.OrderedDict()
    for word in filered_word:
        stem_word = ps.stem(word)
        if stem_word in word_weight:
            word_weight[stem_word] += 1
        else:
            word_weight[stem_word] = 1

    phase_weight = collections.OrderedDict()
    for phase in filtered_phase:
        if phase in phase_weight:
            phase_weight[phase] += 1
        else:
            phase_weight[phase] = 1
    
    return word_weight, phase_weight
    

def search_api(query):
    """
    query:[string] 
    return: list of dict, each dict is a paper record of the original dataset
    """
    # get query word or phase list
    filered_word, filtered_phase = query_to_list(query)
    # create word and phase weight
    word_weight, phase_weight = create_query_weight(filered_word, filtered_phase)   
    # get most cosine-similar 5 document by query weight
    result_ids = search(word_weight, phase_weight)
    return [papers[i] for i in result_ids]

if __name__ == "__main__":
    search_api("knowledge graph")

# initialize the dictionary papers and all word_set
ps = n.stem.PorterStemmer()
stop_words = set(n.corpus.stopwords.words('english'))
# remove the punctuation
tokenizer = n.tokenize.RegexpTokenizer(r'\w+')
# get all abstracts after punctuation removal
papers, abstracts_list = get_abstract_list()
# get abstracts after stemming and inverted index
invert_index = stem_and_index(abstracts_list)
