#!/usr/bin/env python

# Bauné Julie 
# Piter Kenza 
# Tu Justine 

import numpy as np

" 1.a The probability of observing the word wi knowing the two previous words wi-1 and \
wi-2 is P(wi | wi-2, wi-1)"

"1.b Constitution of the tri-grams from sentences' text "


def make_trigramms(text):
    list_trigrams = []
    with open(text) as text:
        text = text.readlines()

        for sentence in text:
            sentence = sentence.split()
            # print(sentence)
            for i in range(0, len(sentence) - 2):
                trigramme = [sentence[i], sentence[i + 1], sentence[i + 2]]
                list_trigrams.append(tuple(trigramme))

    return list_trigrams


# print(make_trigramms("wine.txt"))

# argument de la 2nd fonction
# trigramms = make_trigramms("wine.txt")
# print(len(trigramms))

" 1.c Functions for the constitution of the dictionary that will contain as key the bi-gram of each tri-gram \
 and as value another dictionary containing as key the third word of the tri-gram and as value the probability of \
 the tri-gram to occur."


def make_conditionnal_probas():
    trigramms = make_trigramms("wine.txt")
    # dictionnaire qui contiendra les proba
    counting_table = {}

    # remplissage du dictionnaire avec comme clé les bigrammes de chaque trigramme
    # et en valeur un dictionnaire contenant les mots qui apparaissent après ces
    # bigrammes et combien de fois
    for w in trigramms:
        # print(w)
        bigramme = (w[0], w[1])
        # constitution des bigrammes de chaque trigramme (a et b)
        unigramme = w[2]
        # le mot qui apparait après le bigramme (c)

        # on créé le dictionnaire qui sera en valeur du dictionnaire final counting_table,
        # la clé associé à ce dictionnaire sera le dictionnaire counting table qui contiendra
        # le bigramme en traitement par la boucle
        # si le bigramme n'a jamais été traité on l'ajoute comme clé de counting table
        # et on lui associe un dictionnaire cooc counting vide en valeur
        # sinon on reprend l'entrée du bigramme déjà ajoutée
        cooc_counting = dict(counting_table.setdefault(bigramme, dict()))
        # print(cooc_counting)
        # à l'entrée de counting table sélectionnée précédemment, on ajoute en valeur
        # l'unigramme (c) du bigramme (a, b) et on lui ajoute comme valeur 0 s'il est nouveau
        # puis on ajoute à chaque fois +1 comme ça s'il vient d'être ajouté la valeur de l'occurrence = 1
        # et si il existait déjà son occurrence s'incrémente
        cooc_counting.update({unigramme: cooc_counting.setdefault(unigramme, 0) + 1})
        # print(cooc_counting)
        # une fois que cooc_counting a été rempli pour l'unigramme (c) on met à jour notre dictionnaire counting table
        # en ajoutant la valeur de l'unigramme (c) au bigramme (a, b) que l'on traite
        counting_table.update({bigramme: cooc_counting})

    # calcul des proba
    # pour tous les dico d'unigrammes associés à un bigramme
    for v in counting_table.values():
        # dans le cas ou il y a plusieurs unigrammes (c) pour un bigramme (a, b)
        # on additionne les valeurs de chaque unigramme pour avoir le bon dénominateur
        # car le nombre total de fois que c apparait devant a, b = le nombre de fois que a, b
        # apparait dans le corpus (dénominateur de la formule dans le sujet)
        denominateur = sum(v.values())

        # pour les unigrammes des dico en valeur de counting table
        for k in v:
            # on récupère la valeur de chaque unigramme -> le nombre de fois que le trigamme a, b, c
            # apparait dans le corpus (numérateur de la formule dans le sujet)
            numerateur = v.get(k)
            # print(value)
            # print(v.values())
            # print(len(v.values()))
            # print(numerateur)

            # on met à jour notre dictionnaire contenant le nombre d'occurrence de chaque unigramme (c)
            # pour le remplacer avec la probabilité que ce mot apparaissent après le bigramme a, b
            # donc les clés du dictionnaire
            v.update({k: (numerateur / denominateur)})  # formule de P(c|a,b)

    return counting_table


# ET L'AFFICHAGE DE LA FONCTION !!!!!!!!!!!!!!!!!!!!!!!
# print(make_conditionnal_probas(trigramms))

prob_table = make_conditionnal_probas()

" 1.d The reason why two words have been added to the beginning of each review is \
because we build our language model on trigrams, so to pref-dict the first word of \
a sentence, we need to have two tokens behind. \
As our language model is based on trigrams, considering 1 or 5 tokens or a number different than \
two is not relevant."

"2.1 The history should be initialized with the bi-grams 'BEGIN NOW' because that's how,\
 in the corpus, beginning of sentences are declared."

"This function generates sentences regarding the probabilities of each tri-grams calculated before and for each \
 sentence, calculate the perplexity."


def generate_and_perplexity(prob_table):
    nb = input("How many sentences do you want to generate ? ")

    for i in range(int(nb)):

        history = ["BEGIN", "NOW"]
        # print(history)
        sentence = "BEGIN NOW"
        pp = 1

        mot1 = history[0]
        mot2 = history[1]
        new = ""

        while new != "END":
            # print(mot1, mot2)
            item = prob_table.get((mot1, mot2))
            # print(item)

            words = list(item.keys())  # liste des mots pouvant se trouver après les 2 mots avant
            # print(words)
            probas = list((item.values()))
            # print(probas)

            new = np.random.choice(words, p=probas)
            # print("new =", new)

            p = item.get(new)
            pp *= p
            # print(pp)

            sentence = sentence + " " + new
            # print(sentence)

            mot1, mot2 = mot2, new

        perplexity = pp ** ((-1) / len(sentence))
        print(sentence)
        print(perplexity)


print(generate_and_perplexity(prob_table=prob_table))

"2.2 For a majority of sentences, the grammar and syntax are very good, the use of ponctuation is quite good as well,\
 the commas are correctly placed within the sentence and after a point there is a capital letter. But the semantic is \
 some time kind of weird. The sentence does not make much sense. But overall, the results are quite good."

"2.3 It is possible to estimate the quality of a generated sentence with the measure of \
 the perplexity. It consists of multiplying each trigrams' probability of the sentence and to normalize \
 it with the length of the sentence. We have perplexity = multiplication of the probability of each trigramms to occur \
 to the power of -1 by the length of the sentence generated. The perplexity can be seen "

" the youtube video used for the implementation of the perplexity : \
 https://www.youtube.com/watch?v=GkG-P12B4u0&t=332s "
