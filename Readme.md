# Belgium Traffic Signs Classification
Develop deep learning architectures for classifying Belgium traffic signs. Goals of this project are as follows - 
1. Create structured and automated workflow of classification for reproducible research.
2. Learn and play with convolutional neural networks in TensorFlow


## Download data
1. Download training data - http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip.
2. Download testing data - http://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip
3. './data/download_data.py' will download data and create './data/Training' and './data/Testing'

4. Create empty .gitkeep files (Only if this repo is not being cloned)
echo $null>>.\data\Training\.gitkeep $null>>.\data\Testing\.gitkeep

Git does not store empty directories. .gitkeep enforces directory persistence. 

3. Create .gitignore 
echo $null >> .gitignore
