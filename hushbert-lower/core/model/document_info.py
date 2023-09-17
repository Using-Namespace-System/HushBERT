# hushbert/core/model/document_info.py

"""Module for recursivley reclassifying noise."""

import pandas
import bertopic
  

def get_document_info(model, docs):
    """

    Parameters
    ----------
    model : BERTopic Model
        Topic model.
    docs : Array
        Array of documents to be clasified.

    Returns
    -------
    Pandas DataFrame
        Dataframe with document info and information on recursive layers of classifications.

    """
    limit = 4
    topics, probs = model.fit_transform(docs)
    document_info = model.get_document_info(docs)
    document_info['joining_index'] = document_info.index
    document_info['recursion_layer'] = limit
    return get_document_info_recurs(model, document_info, limit=limit)


def get_document_info_recurs(model, document_info, limit=0, topic=-1):
    """
    
    Parameters
    ----------
    model : BERTopic Model
        Topic model.
    document_info : DataFrame
        Documents and classification information.
    limit : Int, optional
        Depth of recursion. The default is 0.
    topic : Int, optional
        Topic to recursivley reclassify. The default is -1.

    Returns
    -------
    Pandas DataFrame
        Dataframe with document info and information on recursive layers of classifications.

    """
    topic_document_info = document_info.loc[document_info.Topic==topic]
    topic_document_info.recursion_layer = limit - 1
    model.fit_transform(topic_document_info.Document.to_list())
    topic_document_info = model.get_document_info(topic_document_info.Document.to_list(), df=topic_document_info)
    limit = limit - 1
    if(limit>=0 and len(topic_document_info.loc[topic_document_info.Topic==topic]) > 250):
        topic_document_info = get_document_info_recurs(model, topic_document_info, limit=limit)
    topicdocumentIndex = pandas.Index(topic_document_info.joining_index)
    topic_document_info.set_index(topicdocumentIndex)
    document_info.loc[document_info.Topic==topic] = topic_document_info
    return document_info

    