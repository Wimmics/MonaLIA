# MonaLIA

This repository contains the supporting code for the project MonaLIA (2018-2020) sponsored by French Ministry of Culture. The objective of the project is to exploit the crossover between the Machine Learning methods of image analysis and knowledge-based representation and reasoning and its application to the semantic indexing of annotated works and images in Joconde dataset. The goal is to identify automated or semi-automated tasks to improve the annotation and information retrieval.

The code is of prototype quality and supports the experiments conducted during the length of the project. It consists of MonaLIA application to run the PyTorch-based deep learning training written in Python and multiple Jupiter Notebooks for data analysis and visualization.

To be able to run the application and Jupyter Notebooks a local copy of Jococnde dataset (RDF Knowledge Base and images, maintained by French Ministry of Culture) is required as well as CORESE (https://project.inria.fr/corese/download/) or Apache Jena Fuzeki (https://jena.apache.org/documentation/fuseki2/) SPARQL engines to host the Knowledge Base and execute the SPARQL queries. Some queries require CORESE specifically. The dataset and SPARQL engines are not included into this repository.

