import json
import math
import pickle
from contextlib import closing
import nltk
import re
import numpy as np
from google.cloud import storage
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
nltk.download('stopwords')
import inverted_index_gcp
from inverted_index_gcp import InvertedIndex
from collections import Counter
import pandas as pd




class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False



# get all english StopWords and Corpus StopWords

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "make"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this
                     # many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


# initialize bucket access

bucket_name = '302390778ass3'
client = storage.Client()
blobs = client.list_blobs(bucket_name)


# help functions for reading JSON files and posting lists

def read_json(file_route, bucket_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    # get the blob
    blob = bucket.get_blob(file_route)
    # load blob using json
    file = json.loads(blob.download_as_string())
    return file


def read_posting_list(inverted, w,folder, bucket):
  with closing(inverted_index_gcp.MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE,folder, bucket)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list



# getting all the different Indexes from the Bucket (body, title, anchor, normalizedVector)


idx_body = None
for blob in client.list_blobs(bucket_name, prefix="postings_gcp"):
    if blob.name.endswith("pkl"):
        picke_in = blob.download_as_string()
        idx_body = pickle.loads(picke_in)
        break

idx_anchor = None
for blob in client.list_blobs(bucket_name, prefix="postings_anchor"):
    if blob.name.endswith("pkl"):
        picke_in = blob.download_as_string()
        idx_anchor = pickle.loads(picke_in)
        break

idx_title = None
for blob in client.list_blobs(bucket_name, prefix="postings_title"):
    if blob.name.endswith("pkl"):
        picke_in = blob.download_as_string()
        idx_title = pickle.loads(picke_in)
        break


idx_help = None
for blob in client.list_blobs(bucket_name, prefix="data_index"):
    if blob.name.endswith("pkl"):
        picke_in = blob.download_as_string()
        idx_help = pickle.loads(picke_in)
        break


normelized = None
for blob in client.list_blobs(bucket_name, prefix="dfIdf"):
    if blob.name.endswith("pkl"):
        picke_in = blob.download_as_string()
        normelized = pickle.loads(picke_in)
        break



# read JSON files (pageRank and pageVies)

pageRank = read_json('json_files/page_rank.json', bucket_name)
pageViews = read_json('json_files/page_views.json', bucket_name)



# main Search function (search the relevant articles by combination of body, title, anchor and pageRank)

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    finalList = []
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = []
    for tok in tokens:
        if tok in all_stopwords or tok not in idx_body.df.keys():
            continue
        else:
            query.append(tok)
    # BEGIN SOLUTION
    if len(query) == 0:
        return jsonify(res)
    # get rank for body index
    cosSimDic = get_topN_score_for_queries(query, idx_body, 100)  # {q1:[(docId1,rank),(docId2,rank)]}
    body_ranks = list(cosSimDic.values())

    # get rank for title rank
    count = Counter()
    for term in query:
        if term in idx_title.df.keys():
            list_of_doc = read_posting_list(idx_title, term, "postings_title", bucket_name)
            for tup in list_of_doc:
                if tup[0] in count:
                    count[tup[0]] += tup[1]
                else:
                    count[tup[0]] = tup[1]
    title_rank = sorted(count.items(), key=lambda tup: tup[1], reverse=True)[:100]  # [(docId,rank),....]

    # get rank for anchor index
    for term in query:
        if term in idx_anchor.df.keys():
            list_of_doc = read_posting_list(idx_anchor, term, "postings_anchor", bucket_name)
            for tup in list_of_doc:
                if tup[0] in count:
                    count[tup[0]] += tup[1]
                else:
                    count[tup[0]] = tup[1]
    anchor_rank = sorted(count.items(), key=lambda tup: tup[1], reverse=True)[:100]

    #normilazed scores
    normilazed_scores(body_ranks[0], title_rank, anchor_rank)

    # merage between ranks
    wikiID = merge_results(body_ranks[0], title_rank, anchor_rank, title_weight=0.27, text_weight=0.38, anchor_weight=0.35)
    wikiID = [tup[0] for tup in wikiID]
    for id in wikiID:
        finalList.append((id, idx_help.id_title_dic[id]))
    res = finalList
    return jsonify(res)



# body Search function (search the relevant articles by the best similarity between the Query and the Articles by using tf-idf)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    query = []
    for tok in tokens:
        if tok in all_stopwords or tok not in idx_body.df.keys():
            continue
        else:
            query.append(tok)
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    cosSimDic = get_topN_score_for_queries(query, idx_body, 100)  # {q1:[(1,rank),(2,rank)]}
    resQuey = cosSimDic[1]
    finalList = []
    wikiID = [tup[0] for tup in resQuey]
    for id in wikiID:
        finalList.append((id, idx_help.id_title_dic[id]))
    res = finalList
    # END SOLUTION
    return jsonify(res)



# title Search function (search the relevant articles by the best similarity between the Query and the Articles by the number of common words in the titles and the number of their appearances)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    qury = []
    for tok in tokens:
        if tok in all_stopwords or tok not in idx_body.df.keys():
            continue
        else:
            qury.append(tok)
    # BEGIN SOLUTION
    if len(qury) == 0:
        return jsonify(res)
    count = Counter()
    finalList = []
    for term in qury:
        if term in idx_title.df.keys():
            list_of_doc = read_posting_list(idx_title, term, "postings_title", bucket_name)
            for tup in list_of_doc:
                if tup[0] in count:
                    count[tup[0]] += tup[1]
                else:
                    count[tup[0]] = tup[1]
    sortedLst = sorted(count.items(), key=lambda tup: tup[1], reverse=True)[:100]
    for id, freq in sortedLst:
        finalList.append((id, idx_help.id_title_dic[id]))
        finalList.append(id)
    # END SOLUTION
    return jsonify(finalList)



# anchor Search function (search the relevant articles by the best similarity between the Query and the Articles by the number of anchors pointed to the articles with the terms of the query)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    qury = []
    for tok in tokens:
        if tok in all_stopwords or tok not in idx_body.df.keys():
            continue
        else:
            qury.append(tok)
    if len(qury) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    count = Counter()
    finalList = []
    for term in qury:
        if term in idx_anchor.df.keys():
            list_of_doc = read_posting_list(idx_anchor, term, "postings_anchor", bucket_name)
            for tup in list_of_doc:
                if tup[0] in count:
                    count[tup[0]] += tup[1]
                else:
                    count[tup[0]] = tup[1]
    sortedLst = sorted(count.items(), key=lambda tup: tup[1], reverse=True)[:100]
    for id, freq in sortedLst:
        finalList.append((id, idx_help.id_title_dic[id]))
    # END SOLUTION
    return jsonify(finalList)



# pageRank function (return a list of the pageRank values of the given articles ID's)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    pRank = []
    for id in wiki_ids:
        try:
            pRank.append(pageRank[id])
        except:
            continue
    # END SOLUTION
    return jsonify(pRank)



# pageView function (return a list of the pageVies values of the given articles ID's)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    pvRank = []
    for id in wiki_ids:
        try:
            pvRank.append(pageViews[id])
        except:
            continue
    # END SOLUTION
    return jsonify(pvRank)


# help functions for the search functions


# function that calculate the normalized score for each candidate document

def normilazed_scores(body_ranks, title_rank, anchor_rank):
    scores_body = [x[1] for x in body_ranks]
    scores_title = [x[1] for x in title_rank]
    scores_anchor = [x[1] for x in anchor_rank]
    maxBody = max(scores_body)
    maxTitle = max(scores_title)
    maxAnchor = max(scores_anchor)
    minBody = min(scores_body)
    minTitle = min(scores_title)
    minAnchor = min(scores_anchor)
    bRank = []
    for i in range(max(len(body_ranks),len(title_rank),len(anchor_rank))):
        if maxBody > minBody and i < len(scores_body) - 1:
            body_ranks[i] = (body_ranks[i][0],(body_ranks[i][1]-(minBody))/(maxBody-minBody))
        if maxTitle > minTitle and i < len(scores_title) - 1:
            title_rank[i] = (title_rank[i][0], (title_rank[i][1] - (minTitle)) / (maxTitle - minTitle))
        if maxAnchor > minAnchor and i < len(scores_anchor) - 1:
            anchor_rank[i] = (anchor_rank[i][0], (anchor_rank[i][1] - (minAnchor)) / (maxAnchor - minAnchor))


# function that calculate the result of the scores for each candidate document

def merge_results(body_rank, title_rank, anchor_rank, title_weight, text_weight, anchor_weight):
    dic = {}
    pRankDic = {}
    for tup in body_rank:
        dic[tup[0]] = text_weight * tup[1]
        if str(tup[0]) in pageRank.keys():
            pRankDic[str(tup[0])] = pageRank[str(tup[0])]
    for tup in anchor_rank:
        if tup[0] in dic.keys():
            dic[tup[0]] += tup[1] * anchor_weight
            if str(tup[0]) in pageRank.keys():
                pRankDic[str(tup[0])] = pageRank[str(tup[0])]
        else:
            dic[tup[0]] = anchor_weight * tup[1]
            if str(tup[0]) in pageRank.keys():
                pRankDic[str(tup[0])] = pageRank[str(tup[0])]
    for tup in title_rank:
        if tup[0] in dic.keys():
            dic[tup[0]] += tup[1] * title_weight
            if str(tup[0]) in pageRank.keys():
                pRankDic[str(tup[0])] = pageRank[str(tup[0])]
        else:
            dic[tup[0]] = title_weight * tup[1]
            if str(tup[0]) in pageRank.keys():
                pRankDic[str(tup[0])] = pageRank[str(tup[0])]
    maxPr = max(pRankDic.values())
    minPr = min(pRankDic.values())
    if maxPr > minPr:
        for doc_id, score in dic.items():
            if str(doc_id) in pageRank.keys():
                dic[doc_id] = 0.85 * score + (0.15 * (pRankDic[str(doc_id)]-minPr)/(maxPr-minPr))
    return list((sorted(dic.items(), key=lambda tup: tup[1], reverse=True)))[:100]


# function that generates the tf-idf vector for the query

def generate_query_tfidf_vector(query_to_search, index):
    """
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(query_to_search)
    Q = np.zeros((total_vocab_size))
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
        df = index.df[token]
        idf = math.log((len(idx_help.documentLength)) / (df + epsilon), 10)  # smoothing
        try:
            ind = query_to_search.index(token)
            Q[ind] = tf * idf
        except:
            pass
    return Q


# function that gets the candidates documents for the specific query

def get_candidate_documents_and_scores(query_to_search, index):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(idx_help.documentLength)
    for term in np.unique(query_to_search):
        list_of_doc = read_posting_list(idx_body, term, "postings_gcp", '302390778ass3')
        if len(query_to_search) > 2:
            list_of_doc = sorted(list_of_doc, key=lambda x: x[1], reverse=True)[:50]
        else:
            list_of_doc = sorted(list_of_doc, key=lambda x: x[1], reverse=True)[:100]
        for docId, freq in list_of_doc:
            tf = freq / normelized[docId][1]
            idf = math.log(N/idx_body.df[term], 10)
            tfidf = tf*idf
            candidates[(docId, term)] = tfidf
    return candidates


# function that generates the tf-idf vector for each document

def generate_document_tfidf_matrix(query_to_search, index):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(query_to_search)
    # dic = {(docId,term):tfidf}
    candidates_scores = get_candidate_documents_and_scores(query_to_search,
                                                           idx_body)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)
    D.index = unique_candidates
    D.columns = query_to_search

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


# function that calculate the cosine - similarity between the query vector and the documents matrix

def cosine_similarity(D, Q, queryNorm):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    dic = {}
    for id, row in D.iterrows():
        cos_sim = np.dot(Q, row) * (queryNorm * (1/normelized[id][0]))  # need to check
        dic[id] = cos_sim
    return dic


# function that calculate the Normal of the query vector

def getQueryNorm(lst):
    lenOfDoc = len(lst)
    count = Counter(lst)
    normTfIdf = 0
    for word in count:
        normTfIdf += count[word] ** 2
    if normTfIdf == 0:
        return 0.00000001
    else:
        normTfIdf = (1 / (normTfIdf ** (0.5)))
    return normTfIdf


# function that get the top n Document by their scores

def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))
    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                  reverse=True)[:N]


def get_topN_score_for_queries(query, index, N=3):
    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    dic = {}
    Q = generate_query_tfidf_vector(query, index)
    D = generate_document_tfidf_matrix(query, index)
    queryNorm = getQueryNorm(query)
    sim_dict = cosine_similarity(D, Q, queryNorm)
    dic[1] = get_top_n(sim_dict, N)
    return dic




if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
    #checkQueryIndex()
