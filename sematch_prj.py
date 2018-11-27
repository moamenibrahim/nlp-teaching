# Computing Word Similarity
from sematch.semantic.similarity import WordNetSimilarity
wns = WordNetSimilarity()

# Computing English word similarity using Li method
wns.word_similarity('dog', 'cat', 'li') # 0.449327301063
# Computing Spanish word similarity using Lin method
wns.monol_word_similarity('perro', 'gato', 'spa', 'lin') #0.876800984373
# Computing Chinese word similarity using  Wu & Palmer method
wns.monol_word_similarity('狗', '猫', 'cmn', 'wup') # 0.857142857143
# Computing Spanish and English word similarity using Resnik method
wns.crossl_word_similarity('perro', 'cat', 'spa', 'eng', 'res') #7.91166650904
# Computing Spanish and Chinese word similarity using Jiang & Conrad method
wns.crossl_word_similarity('perro', '猫', 'spa', 'cmn', 'jcn') #0.31023804699
# Computing Chinese and English word similarity using WPath method
wns.crossl_word_similarity('狗', 'cat', 'cmn', 'eng', 'wpath')#0.593666388463

# Computing semantic similarity of YAGO concepts.
from sematch.semantic.similarity import YagoTypeSimilarity
sim = YagoTypeSimilarity()

#Measuring YAGO concept similarity through WordNet taxonomy and corpus based information content
sim.yago_similarity('http://dbpedia.org/class/yago/Dancer109989502','http://dbpedia.org/class/yago/Actor109765278', 'wpath') #0.642
sim.yago_similarity('http://dbpedia.org/class/yago/Dancer109989502','http://dbpedia.org/class/yago/Singer110599806', 'wpath') #0.544
#Measuring YAGO concept similarity based on graph-based IC
sim.yago_similarity('http://dbpedia.org/class/yago/Dancer109989502','http://dbpedia.org/class/yago/Actor109765278', 'wpath_graph') #0.423
sim.yago_similarity('http://dbpedia.org/class/yago/Dancer109989502','http://dbpedia.org/class/yago/Singer110599806', 'wpath_graph') #0.328

# Computing semantic similarity of DBpedia concepts.
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
concept = ConceptSimilarity(Taxonomy(DBpediaDataTransform()),'models/dbpedia_type_ic.txt')
concept.name2concept('actor')
concept.similarity('http://dbpedia.org/ontology/Actor','http://dbpedia.org/ontology/Film', 'path')
concept.similarity('http://dbpedia.org/ontology/Actor','http://dbpedia.org/ontology/Film', 'wup')
concept.similarity('http://dbpedia.org/ontology/Actor','http://dbpedia.org/ontology/Film', 'li')
concept.similarity('http://dbpedia.org/ontology/Actor','http://dbpedia.org/ontology/Film', 'res')
concept.similarity('http://dbpedia.org/ontology/Actor','http://dbpedia.org/ontology/Film', 'lin')
concept.similarity('http://dbpedia.org/ontology/Actor','http://dbpedia.org/ontology/Film', 'jcn')
concept.similarity('http://dbpedia.org/ontology/Actor','http://dbpedia.org/ontology/Film', 'wpath')

# Computing semantic similarity of DBpedia entities.
from sematch.semantic.similarity import EntitySimilarity
sim = EntitySimilarity()
sim.similarity('http://dbpedia.org/resource/Madrid','http://dbpedia.org/resource/Barcelona') #0.409923677282
sim.similarity('http://dbpedia.org/resource/Apple_Inc.','http://dbpedia.org/resource/Steve_Jobs')#0.0904545454545
sim.relatedness('http://dbpedia.org/resource/Madrid','http://dbpedia.org/resource/Barcelona')#0.457984139871
sim.relatedness('http://dbpedia.org/resource/Apple_Inc.','http://dbpedia.org/resource/Steve_Jobs')#0.465991132787

# Evaluate semantic similarity metrics with word similarity datasets
from sematch.evaluation import WordSimEvaluation
from sematch.semantic.similarity import WordNetSimilarity
evaluation = WordSimEvaluation()
evaluation.dataset_names()
wns = WordNetSimilarity()
# define similarity metrics
wpath = lambda x, y: wns.word_similarity_wpath(x, y, 0.8)
# evaluate similarity metrics with SimLex dataset
evaluation.evaluate_metric('wpath', wpath, 'noun_simlex')
# performa Steiger's Z significance Test
evaluation.statistical_test('wpath', 'path', 'noun_simlex')
# define similarity metrics for Spanish words
wpath_es = lambda x, y: wns.monol_word_similarity(x, y, 'spa', 'path')
# define cross-lingual similarity metrics for English-Spanish
wpath_en_es = lambda x, y: wns.crossl_word_similarity(x, y, 'eng', 'spa', 'wpath')
# evaluate metrics in multilingual word similarity datasets
evaluation.evaluate_metric('wpath_es', wpath_es, 'rg65_spanish')
evaluation.evaluate_metric('wpath_en_es', wpath_en_es, 'rg65_EN-ES')


# Evaluate semantic similarity metrics with category classification
from sematch.evaluation import AspectEvaluation
from sematch.application import SimClassifier, SimSVMClassifier
from sematch.semantic.similarity import WordNetSimilarity

# create aspect classification evaluation
evaluation = AspectEvaluation()
# load the dataset
X, y = evaluation.load_dataset()
# define word similarity function
wns = WordNetSimilarity()
word_sim = lambda x, y: wns.word_similarity(x, y)
# Train and evaluate metrics with unsupervised classification model
simclassifier = SimClassifier.train(zip(X,y), word_sim)
evaluation.evaluate(X,y, simclassifier)


# Matching Entities with type using SPARQL queries
from sematch.application import Matcher
matcher = Matcher()
# matching scientist entities from DBpedia
matcher.match_type('scientist')
matcher.match_type('científico', 'spa')
matcher.match_type('科学家', 'cmn')
matcher.match_entity_type('movies with Tom Cruise')

# Entity feature extraction with Similarity Graph
from sematch.semantic.graph import SimGraph
from sematch.semantic.similarity import WordNetSimilarity
from sematch.nlp import Extraction, word_process
from sematch.semantic.sparql import EntityFeatures
from collections import Counter
tom = EntityFeatures().features('http://dbpedia.org/resource/Tom_Cruise')
words = Extraction().extract_nouns(tom['abstract'])
words = word_process(words)
wns = WordNetSimilarity()
word_graph = SimGraph(words, wns.word_similarity)
word_scores = word_graph.page_rank()
words, scores =zip(*Counter(word_scores).most_common(10))
print(words)