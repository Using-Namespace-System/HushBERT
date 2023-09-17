# HushBERT/core/model/document_info.py

"""Module for recursivley reclassifying noise."""

import pandas as pd
import bertopic as BERTopic
  

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
    document_info = model.get_document_info(docs)
    return get_document_info_recurs(model, docs, document_info, limit=3)


def get_document_info_recurs(model, document_info, limit, topic=-1):
    """
    
    Parameters
    ----------
    model : BERTopic Model
        Topic model.
    document_info : DataFrame
        Documents and classification information.
    limit : Int
        Depth of recursion.
    topic : Int, optional
        Topic to recursivley reclassify. The default is -1.

    Returns
    -------
    Pandas DataFrame
        Dataframe with document info and information on recursive layers of classifications.

    """
    document_info['recursion_layer'] = limit + 1
    document_info['joining_index'] = document_info.index
    topic_document_info = document_info.loc[document_info.Topic==topic]
    model.fit_transform(topic_document_info.Document.to_list())
    model.get_document_info(topic_document_info.Document.to_list(), df=topic_document_info)
    if(limit>=0):
        limit = limit - 1
        topic_document_info_recurs = get_document_info_recurs(model, topic_document_info, limit)
        topic_document_info_recurs['primary_joining_index'] = topic_document_info_recurs['joining_index']
        topic_document_info['primary_joining_index'] = topic_document_info['joining_index']
        topic_document_info_recurs.set_index('primary_joining_index')
        topic_document_info.set_index('primary_joining_index')
        topic_document_info.loc[topic_document_info.Topic==topic] = topic_document_info_recurs
        return topic_document_info
    else:
        topic_document_info['primary_joining_index'] = topic_document_info['joining_index']
        document_info['primary_joining_index'] = document_info['joining_index']
        topic_document_info.set_index('primary_joining_index')
        document_info.set_index('primary_joining_index')
        document_info.loc[document_info.Topic==topic] = topic_document_info
        return document_info

    