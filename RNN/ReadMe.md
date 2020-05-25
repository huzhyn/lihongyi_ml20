## RNN 作业
本次作业是要让同学接触 NLP 当中一个简单的 task —— 语句分类（文本分类）
给定一个语句，判断他有没有恶意（负面标 1，正面标 0）

有三个档案，分别是 training_label.txt、training_nolabel.txt、testing_data.txt

training_label.txt：有 label 的 training data（句子配上 0 or 1，+++$+++ 只是分隔符号，不要理它）

e.g., 1 +++$+++ are wtf ... awww thanks !
training_nolabel.txt：没有 label 的 training data（只有句子），用来做 semi-supervised learning

ex: hates being this burnt !! ouch
testing_data.txt：你要判断 testing data 裡面的句子是 0 or 1

id,text

0,my dog ate our dinner . no , seriously ... he ate it .

1,omg last day sooon n of primary noooooo x im gona be swimming out of school wif the amount of tears am gona cry

2,stupid boys .. they ' re so .. stupid !


## 下载数据
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1dPHIl8ZnfDz_fxNd2ZeBYedTat2lfxcO' -O 'drive/My Drive/Colab Notebooks/hw8-RNN/data/training_label.txt'
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1x1rJOX_ETqnOZjdMAbEE2pqIjRNa8xcc' -O 'drive/My Drive/Colab Notebooks/hw8-RNN/data/training_nolabel.txt'
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=16CtnQwSDCob9xmm6EdHHR7PNFNiOrQ30' -O 'drive/My Drive/Colab Notebooks/hw8-RNN/data/testing_data.txt'

上面3行命令与下面命令的作用是一样的，都是下载数据
gdown --id '1lz0Wtwxsh5YCPdqQ3E3l_nbfJT1N13V8' --output data.zip
unzip data.zip
