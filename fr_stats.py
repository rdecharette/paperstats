import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
from tqdm import tqdm
import Levenshtein as lev
from unidecode import unidecode

conf = "cvpr/cvpr2024.json"

def string_preprocess(list):
    return np.strings.strip([np.char.title(unidecode(s)) for s in list], "., ")

fr_institutes = None
def is_french(list):
    global fr_institutes
    list = np.asarray(list)
    if fr_institutes is None:
        # ChatGPT prompt used to retrieve json institutes:
        # can you list me all the universities, research institutes and companies in France that publish in the field of AI or computer vision ? Please reply in french, and list all names in a json array. Just include the name, nothing more. If an institute name exist in different variantes, for example "ENS" and "Ecole normale supérieure", please include all variants separately. Do not include duplicate names. Do not include names of institutes that sound french but are in fact in Québec or other French speaking countries.
        fr_institutes = string_preprocess(json.load(open('fr_institutes.json')))

    french_mask = np.zeros_like(list, dtype=bool)

    keywords = ["France", "Paris", "Marseille", "Lyon", "Toulouse", "Nice", "Nantes", "Strasbourg", "Montpellier", "Bordeaux", "Lille"]
    keywords = string_preprocess(keywords)
    keywords_mask = np.vstack([np.strings.find(list, k) > -1 for k in keywords])

    french_mask = np.any(np.vstack([french_mask, keywords_mask]), axis=0)

    institute_mask = np.zeros_like(french_mask)
    print("Find FR institutes using levenshtein distance")
    for i, inst in enumerate(tqdm(list)):
        if french_mask[i]:
            continue
        for j, fr_inst in enumerate(fr_institutes):
            dist = lev.distance(inst, fr_inst)
            dist_n = dist / max(len(inst), len(fr_inst))
            # print(inst, "\t", fr_inst, ": \t", dist, "\t", dist_n)
            if dist_n < 0.2:
                # print("BINGO")
                institute_mask[i] = True
                break

    french_mask = np.any(np.vstack([french_mask, institute_mask]), axis=0)
    return french_mask

# Debug
# insts = ["INRIA", "Inria", "Inrie", "Université Paris", "Carnegie Mellon"]
# fr_m = is_french(insts)


conf_path = os.path.join('thirdparty','paperlists',conf)
f = open(conf_path)
data = np.asarray(json.load(f))



print("Prepare data")
# Split affiliations into array
for i, paper in enumerate(tqdm(data)):
    aff = paper["aff"]
    # aff = re.sub("(?i)university", "uni.", paper["aff"])
    # aff = re.sub("(?i)institute", "inst.", paper["aff"])
    paper["authors"] = string_preprocess(paper["author"].split(", "))
    paper["affiliations"] = string_preprocess(np.asarray(aff.split("; ")))
    paper["affiliations"] = paper["affiliations"][paper["affiliations"]!=""]

aff_all = np.hstack([paper["affiliations"] for paper in data])
aff_u, aff_u_c = np.unique(aff_all, return_counts=True)
aff_stats = d = {**dict(zip(aff_u, aff_u_c))}



print("Count paper affiliations")
aff_u_papers = np.zeros(aff_u.shape)
for i, aff in enumerate(tqdm(aff_u)):
    aff_u_papers[i] = np.sum([np.any(paper["affiliations"]==aff) for paper in data])

# Debug snippet
# aff_u[5], np.argwhere([np.any(paper["affiliations"]==aff_u[5]) for paper in data]), [p[0]["aff"] for p in np.array(data)[np.argwhere([np.any(paper["affiliations"]==aff_u[5]) for paper in data])]]

plt.figure()
s = aff_u_papers.argsort()[::-1][:25]
sns.barplot(aff_u_papers[s], edgecolor="black")
plt.xticks(range(len(aff_u_papers[s])), [s if len(s) < 35+5 else s[:35]+" (..)" for s in aff_u[s]], rotation=45, ha='right')

plt.ylabel("Count")
plt.title("%d Affiliations\n(a paper can have multiple affiliations)"%(sum(aff_u_papers)))
plt.tight_layout()
plt.savefig("papers_affiliations.png")



print("Compute FR affiliations")
fr_mask = is_french(aff_u)
aff_fr_u = aff_u[fr_mask]
aff_fr_u_c = aff_u_c[fr_mask]

print("Found FR affiliations (%d): "%(len(aff_fr_u)), aff_fr_u)



print("French affiliations")
aff_fr_u_papers = np.zeros(aff_fr_u.shape)
for i, aff in enumerate(tqdm(aff_fr_u)):
    aff_fr_u_papers[i] = np.sum([np.any(paper["affiliations"]==aff) for paper in data])

plt.figure()
s = aff_fr_u_papers.argsort()[::-1][:25]
sns.barplot(aff_fr_u_papers[s], edgecolor="black")
plt.xticks(range(len(aff_fr_u_papers[s])), [s if len(s) < 35+5 else s[:35]+" (..)" for s in aff_fr_u[s]], rotation=45, ha='right')

plt.ylabel("Count")
plt.title("%d French affiliations\n(a paper can have multiple FR affiliations)"%(sum(aff_fr_u_papers)))
plt.tight_layout()
plt.savefig("fr_papers_affiliations.png")

print("Count paper at least 1 French affiliation")
data_fr = np.zeros(len(data), dtype=int)
for i, paper in enumerate(tqdm(data)):
    data_fr[i] = np.sum([np.any(paper["affiliations"]==a_fr) for a_fr in aff_fr_u])

data_fr_authors = np.unique(np.hstack([paper["authors"] for paper in data[data_fr>0]]))

print("Total affiliations: ", sum(data_fr))
print("Paper w/ at least 1 FR affiliation: ", sum(data_fr>0))
print("     # of authors: ", len(data_fr_authors))
print("     authors: ", data_fr_authors)

plt.show()