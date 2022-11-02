import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
import scipy.cluster.hierarchy as scp
#plt.rcParams.update({'font.size': 8})

# https://www.sbert.net/docs/pretrained_models.html

text_model = SentenceTransformer('all-mpnet-base-v2') # all-mpnet-base-v2, all-distilroberta-v1, all-MiniLM-L6-v2

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def embed_compare(sentence):
    text_embedding = text_model.encode(sentence)
    sim_mat = np.inner(text_embedding,text_embedding)
    return sim_mat

def plot_sim(sim_mat, labels):
    im = plt.imshow(sim_mat)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title('Semantic similarity')
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(i, j, np.round(sim_mat[i, j], 2), color='black', ha='center', va='center', fontsize=8)

words = ['An','All','My','Your','This','That','These','Those']
sim_word = embed_compare(words)

phrases = ['An apple','All apple','My apple','Your apple','This apple','That apple','These apple','Those apple']
sim_phrase = embed_compare(phrases)


f = plt.figure(figsize=(10,8))
plt.suptitle('Sentence BERT')
plt.subplot(221)
plot_sim(sim_word, words)
plt.subplot(222)
plot_sim(sim_phrase, phrases)

plt.subplot(223)
Zw = scp.linkage(1-sim_word)
dendrow = scp.dendrogram(Zw, labels=words,leaf_font_size=8, leaf_rotation=90)

plt.subplot(224)
Zp = scp.linkage(1-sim_phrase)
dendrop = scp.dendrogram(Zp, labels=phrases,leaf_font_size=8, leaf_rotation=90)

plt.tight_layout()
plt.show()

f.savefig('../Fig/sBERT_analysis.png')
f.savefig('../Fig/sBERT_analysis.svg')


nouns = ['apple','apples','onion','onions','carrot','carrots','orange','oranges']
sim_nouns = embed_compare(nouns)
Zn = scp.linkage(1-sim_nouns)
f2 = plt.figure()
plt.suptitle('Sentence BERT')
plt.subplot(121)
plot_sim(sim_nouns, nouns)
plt.subplot(122)
dendron = scp.dendrogram(Zn, labels=nouns,leaf_font_size=8, leaf_rotation=90)
plt.tight_layout()
plt.show()
f2.savefig('../Fig/sBERT_noun_analysis.png')
f2.savefig('../Fig/sBERT_noun_analysis.svg')