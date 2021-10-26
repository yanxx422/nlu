
0. Paper
The file AgreementDataset.tsv contains the dataset presented in the paper:
"Agreeing to Disagree: Annotating Offensive Language Datasets with Annotators’ Disagreement. Elisa Leonardelli, Stefano Menini, Alessio Palmero Aprosio, Marco Guerini and Sara Tonelli, in Proceedings of EMNLP 2021"

1. Download
Download the paper at https://arxiv.org/abs/2109.13563
Request the dataset at: https://github.com/dhfbk/annotators-agreement-dataset

2. Authors
Elisa Leonardelli (*)
Stefano Menini (*)
Alessio Palmero Aprosio (*)
Marco Guerini (*)
Sara Tonelli (*)

(*) Fondazione Bruno Kessler

3. Introduction
The agreement dataset consists of 10753 tweets from three different domains (Covid-19, BLM, US elections). Each tweet has been annotated as offensive/not-offensive by 5 different annotators through Amazon Mechanical Turk. More details on how the tweets have been collected and annotated can be found in the paper. 
For each tweet, we release its ID. The original tweets can be retrieved through functions such as:
https://github.com/twitterdev/getting-started-with-the-twitter-api-v2-for-academic-research/blob/main/labs-code/python/academic-research-product-track/tweets_lookup.py#L10


4. File Format:
The dataset file is in a tab-separeted format (.tsv). 
Columns are the following:

- Column 1 "Offensive_binary_label" -->  1: offensive,  0: not offensive. Binary label assigned accordingly to the majority of annotations received.
- Column 2 "Tweet_ID" --> ID of the tweet, to be used to retrive it.
- Column 3 "Crowd_agreement_level" --> level of agreement between the crowd of 5 annotators, regardless of the label. A++: 5/5 annotators agreed on the same label, A+: 4/5 agreed on the same label, A0: 3/5 agreed on the same label. 
- Column 4 "Crowd_annotations" --> level of agreement per type of label. N++: 5/5 annotations not offensive, N+: 4/5 not offensive (1/5 offensive), N0: 3/5 not offensive (2/5 offensive), O++: 5/5 annotations offensive, O+: 4/5 offensive (1/5 not offensive), O0: 3/5 offensive (2/5 not offensive).
- Column 5 "Domain" --> BLM, Elections2020, Covid-19
- Column 6 "Split" --> whether the tweet was used in the experiments presented in the paper ("train", "dev", test") or not ("not used"). The "not-used" items were excluded from the experiments because we wanted to balance our data in terms of agreement/label/domain, but they have been extracted and annotated following exactly the same process as the used ones.


5. Citation 

If you use this dataset please cite:

@inproceedings{leonardelli2021agreeing,
      title={Agreeing to Disagree: Annotating Offensive Language Datasets with Annotators' Disagreement}, 
      author={Leonardelli, Elisa and Menini, Stefano and Palmero Aprosio, Alessio and Guerini, Marco and Tonelli, Sara},
      year={2021},
      publisher ={Proceedings of Empirical Methods in Natural Language Processing 2021 (EMNLP2021)},
      address ={Punta Cana, Dominican Republic},
}
