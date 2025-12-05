## 🏥 Optical Coherence Tomography (OCT) — Introduction

L’Optical Coherence Tomography (OCT) est une technique d’imagerie médicale non invasive utilisée pour obtenir des images détaillées de la rétine.
Grâce à sa haute résolution, l’OCT est aujourd’hui un outil essentiel en ophtalmologie pour le diagnostic, le suivi et la prise en charge des maladies oculaires.

## 👁️ Qu’est-ce que l’OCT ?

L’OCT fonctionne comme une échographie utilisant la lumière :

- un faisceau lumineux traverse l’œil,

- la lumière réfléchie par les différentes couches de la rétine est capturée,

- une image en coupe (B-scan) est reconstruite.

Ce procédé permet de visualiser les structures microscopiques de la rétine avec une précision de quelques microns.

## 🩺 Pathologies détectables en OCT

Les images OCT permettent d’identifier plusieurs anomalies rétiniennes majeures, notamment :

🔹 Choroidal Neovascularization (CNV)

Croissance anormale de vaisseaux sanguins sous la rétine, souvent associée à la DMLA.

🔹 Diabetic Macular Edema (DME)

Accumulation de liquide dans la macula chez les patients diabétiques.

🔹 Drusen (AMD-related)

Dépôts sous-rétiniens visibles dans la dégénérescence maculaire liée à l’âge.

🔹 Normal Retina

Rétine saine, avec des couches régulières sans anomalies.

## 🖼️ Types d’images OCT

Les images OCT peuvent être :

- B-scan : coupe transversale (le format le plus courant)

- Volume scan : ensemble de plusieurs B-scans

- En-face : vue en surface d’une couche rétinienne

Le dataset utilisé dans ce projet contient principalement des B-scans.

## 📚 Dataset OCT (Kermany2018 / OCT2017)

Pour l'analyse et l'entraînement de modèles IA, un dataset OCT public est communément utilisé :
OCT2017 (Kermany2018).

Il contient quatre classes :

- CNV

- DME

- DRUSEN

- NORMAL

Ce dataset est l’un des plus utilisés dans la recherche pour entraîner des modèles de détection automatique de pathologies rétiniennes.

## 🎯 Objectif du projet

Ce repository vise à :

- représenter les images OCT et leur importance clinique,

- comprendre les caractéristiques visuelles permettant d’identifier une pathologie,

- utiliser l’OCT comme base pour le développement de modèles d’IA en imagerie médicale.

## 🧬 Conclusion

L’OCT est aujourd’hui la méthode la plus avancée pour visualiser la rétine en profondeur.
Combinée à l’intelligence artificielle, elle permet de créer des outils puissants pour le diagnostic assisté et le dépistage précoce des maladies oculaires.
