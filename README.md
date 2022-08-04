
# Prédiction de la race de chien à l'aide de CNN et apprentissage par transfert
## Objectifs :
1- Pré-processer les images avec des techniques spécifiques (e.g. whitening, equalization, éventuellement modification de la taille des images).

2- Réaliser de la data augmentation (mirroring, cropping...).

3 - Mise en œuvre de 2 approches de l'utilisation des CNN :
- Réaliser un réseau de neurones CNN from scratch en optimisant les paramètres.
- Utiliser le transfert learning et ainsi utiliser un réseau déjà entrainé.
    
4- Développer une application Web ou mobile pour traiter des images de chien fournies par l'utilisateur. Étant donné une image d'un chien, notre algorithme identifie une estimation de la race du chien.

![Dog prediction](https://github.com/HoudCa/Dog_Breed_Prediction/blob/main/Img_Dog_prediction.png)

## Données
L'ensemble de données [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) contient des images de 120 races de chiens du monde entier. Cet ensemble de 
données a été construit à l'aide d'images et d'annotations d'ImageNet pour la tâche de catégorisation fine des images. Il a été initialement collecté pour la 
catégorisation des images à grain fin, un problème difficile car certaines races de chiens ont des caractéristiques presque identiques ou diffèrent par la couleur et 
l'âge.

- Nombre de catégories : 120
- Nombre d'images : 20 580
- Annotations : étiquettes de classe, boîtes englobantes

*Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual 
Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011*

## Dépendances
Ce projet utilise principalement Python Jupyter Notebooks, OpenCV, tensorflow, Keras, sklearn, PIL
## Instruction
Téléchargez le Notebook et importez-le de préférence dans Google Colaboratoty, Kaggle ou Jupyter via Anaconda.
Ensuite, effectuez simplement "Run all" pour exécuter le projet.


