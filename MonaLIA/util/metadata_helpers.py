# -*- coding: utf-8 -*-
"""
MonaLIA project

Library of helper functions

@author: abobasheva
"""
import pandas as pd
#from itertools import compress

import json
from SPARQLWrapper import SPARQLWrapper, JSON, XML
from SPARQLWrapper import POST, POSTDIRECTLY
from SPARQLWrapper import RDFXML

from rdflib import Graph, URIRef
#from rdflib import RDF, RDFS, XSD
from rdflib.namespace import SKOS
from rdflib.namespace import Namespace, NamespaceManager
from rdflib.plugins import sparql as SPARQL


###############################################################################
# Test
###############################################################################
def read_flat_file_to_df(filePath , sep, show=False):
    df = pd.read_csv(filePath, sep)
    if (show):
        print(df.shape)
        with pd.option_context('display.max_rows', 5, 'display.max_columns', 5):
            print(df)
    return df 

###############################################################################
# RDF and SPARQL
###############################################################################
jcl  = Namespace('http://jocondelab.iri-research.org/ns/jocondelab/')
notice = Namespace("https://jocondelab.iri-research.org/data/notice/")
monalia = Namespace("http://ns.inria.fr/monalia/")
thesaurus = Namespace("http://data.culture.fr/thesaurus/resource/ark:/67717/")


def sparql_service_to_dataframe(service, query):
    """
    Helper function to convert SPARQL results into a Pandas data frame.
    
    Credit to Ted Lawless https://lawlesst.github.io/notebook/sparql-dataframe.html
    """
    sparql = SPARQLWrapper(service)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query()

    processed_results = json.load(result.response)
    cols = processed_results['head']['vars']

    out = []
    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)

    return pd.DataFrame(out, columns=cols)

def sparql_graph_to_dataframe(graph, query):
    """
    Helper function to convert RDFLib graph SPARQL results into a Pandas data frame.
    """
    q =  SPARQL.prepareQuery(query, initNs = { 'monalia': monalia,
                                               'skos': SKOS,
                                               'jcl': jcl })

    res = graph.query(q) 
   
    print('query returned %d entries' % len(res))

    cols = pd.Series(res.vars).apply(str).values

    out = []
    for row in res:
        item = []
        for col in cols:
            item.append(str(row[col]))
        out.append(item)

    return pd.DataFrame(out, columns=cols)

def getJocondeTermByLabel_thesaurus_graph(graph, label):
    """
    Helper function to get a jcl:Term specified by prefered label 
    """
    q = ( '''
        prefix skos: <http://www.w3.org/2004/02/skos/core#>  
        select  ?term  
            where {
            ?term skos:prefLabel "%s"@fr.
           }''' % label)
    
    res = graph.query(q)
    
    for row in res:
        return URIRef(row['term'])

    return None

def getJocondeTermByLabel_service(service, label):
    """
    Helper function to get a jcl:Term specified by prefered label 
    """
    q = ( '''
        prefix skos: <http://www.w3.org/2004/02/skos/core#>  
        select  ?term  
            where {
            ?term skos:prefLabel "%s"@fr.
           }''' % label)
    
    sparql = SPARQLWrapper(service)
    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    result = sparql.query()
    
    res = json.load(result.response)
    #return res
       
    for row in res['results']['bindings']:
        if row['term']['type'] == 'uri':
            return URIRef(row['term']['value'])

    return None


def describeJocondeNotice_service(service, noticeRef):
    """
    Helper function to fetch all the triples for the jcl:Notice
    """
    query_describe = ('''
    prefix skos: <http://www.w3.org/2004/02/skos/core#> 
    prefix jcl: <http://jocondelab.iri-research.org/ns/jocondelab/>
    
    describe <%s> ''' % notice[noticeRef])

    #print(query_describe )
    sparql = SPARQLWrapper(service)
    sparql.setQuery(query_describe)
    sparql.setReturnFormat(XML)
    results = sparql.query().convert()

    return results

def describeJocondeNoticeList_service(service, noticeList):
    """
    Helper function to fetch all the triples for the jcl:Notice
    """
    notice_list_str = ' '.join('"'+ r + '"' for r in noticeList)
    
    query_describe = ('''
    prefix jcl: <http://jocondelab.iri-research.org/ns/jocondelab/>

    describe ?x where {
  
        VALUES ?v {%s}.
        ?x jcl:noticeRef ?v.
    }
    
    ''' % notice_list_str)

    #print(query_describe )
    sparql = SPARQLWrapper(service)
    sparql.setQuery(query_describe)
    sparql.setReturnFormat(XML)
    results = sparql.query().convert()

    return results

def selectJocondeNoticeList_service(service, query_select, noticeList):
    """
    Helper function to fetch all the triples for the jcl:Notice
    """
    notice_list_str = ' '.join('"'+ r + '"' for r in noticeList)
    
    query_select = query_select % notice_list_str

    #print(query_describe )
    sparql = SPARQLWrapper(service)
    sparql.setQuery(query_select)
    sparql.setReturnFormat(XML)
    results = sparql.query().convert()

    return results

def create_graph(name='MonaLIA'):
    g = Graph(identifier=name)

    g.namespace_manager = NamespaceManager(Graph())
    g.namespace_manager.bind('jcl', jcl, override=False)
    g.namespace_manager.bind('ml', monalia, override=False)
    g.namespace_manager.bind('skos', SKOS,  override=False)
    g.namespace_manager.bind('t', thesaurus,  override=False)
    g.namespace_manager.bind('n', notice,  override=False)
    
    return g
    

def sparql_service_update(service, update_query):
    """
    Helper function to update (DELETE DATA, INSERT DATA, DELETE/INSERT) data.
 
    """
    sparql = SPARQLWrapper(service)
    sparql.setMethod(POST)
    sparql.setRequestMethod(POSTDIRECTLY)
    sparql.setQuery(update_query)
    result = sparql.query()
    
    #SPARQLWrapper is going to throw an exception if result.response.status != 200:
                
    return 'Done'
        
def sparql_service_construct(service, contruct_query):
    """
    Helper function to run CONSTRUCT query and 
    return rdflib.graph.ConjunctiveGraph that can be serialized to
    a string or file like in example below
    
    results.serialize(destination='dest_file.ttl' ,
                  format='n3',
                  encoding='utf-8')
 
    """
    sparql = SPARQLWrapper(service)
    sparql.setQuery(contruct_query)
    sparql.setReturnFormat(RDFXML)
    
    results = sparql.query().convert()
    
    return results