---
layout: post
title: Kaggle Competition Writeup - IEEE CIS Fraud Detection
description: "This is a write-up of a presentation on generating music in the waveform domain, which was part of a tutorial that I co-presented at ISMIR 2019 earlier this month."

tags: [machine learning, kaggle competition]

image:
  feature:
comments: false
share: true
---

During this period of lockdown I decided to attempt the IEEE CIS Fraud Detection Kaggle Competition. The ideas that I used for this competition were very well received, so I've decided to write it up in the form of a blog post.

## <a name="overview"></a> Overview

This blog post is divided into a few different sections. Finally, I'll raise some observations and discussion points. If you want to skip ahead, just click the section title below to go there.

* *[Problem Statement](#motivation)*
* *[Dataset](#dataset)*
* *[Exploration](#exploration)*
* *[Feature Engineering](#feature-engineering)*
* *[Validation Strategy](#validation-strategy)*
* *[Models](#models)*
* *[Conclusion](#conclusion)*

Note that this blog post is not intended to provide an exhaustive overview of all the published research in this domain -- I have tried to make a selection and I've inevitably left out some great work. **Please don't hesitate to suggest relevant work in the comments section!**


## <a name="motivation"></a> Problem Statement

In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.

The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

## <a name="dataset"></a> Dataset

The description of the dataset provided in the competitions page was very brief. Below is the more detailed version of the dataset which was present in the discussions section of the competition.

* Transaction table
“It contains money transfer and also other gifting goods and service, like you booked a ticket for others, etc.”

* TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
“TransactionDT first value is 86400, which corresponds to the number of seconds in a day (60 * 60 * 24 = 86400) so I think the unit is seconds. Using this, we know the data spans 6 months, as the maximum value is 15811131, which would correspond to day 183.”

* TransactionAMT: transaction payment amount in USD
“Some of the transaction amounts have three decimal places to the right of the decimal point. There seems to be a link to three decimal places and a blank addr1 and addr2 field. Is it possible that these are foreign transactions and that, for example, the 75.887 in row 12 is the result of multiplying a foreign currency amount by an exchange rate?”

* ProductCD: product code, the product for each transaction
“Product isn't necessary to be a real 'product' (like one item to be added to the shopping cart). It could be any kind of service.”

* card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.

* addr: address
“both addresses are for purchaser
addr1 as billing region
addr2 as billing country”

* dist: distance
"distances between (not limited) billing address, mailing address, zip code, IP address, phone area, etc.”

* P_ and (R_) emaildomain: purchaser and recipient email domain “ certain transactions don't need recipient, so Remaildomain is null.”

* C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
“Can you please give more examples of counts in the variables C1-15? Would these be like counts of phone numbers, email addresses, names associated with the user? I can't think of 15.
Your guess is good, plus like device, ipaddr, billingaddr, etc. Also these are for both purchaser and recipient, which doubles the number.”

* D1-D15: timedelta, such as days between previous transaction, etc.

* M1-M9: match, such as names on card and address, etc.

* Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.
“For example, how many times the payment card associated with a IP and email or address appeared in 24 hours time range, etc.”
"All Vesta features were derived as numerical. some of them are count of orders within a clustering, a time-period or condition, so the value is finite and has ordering (or ranking). I wouldn't recommend to treat any of them as categorical. If any of them resulted in binary by chance, it maybe worth trying."

* Identity Table
Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions.
They're collected by Vesta’s fraud protection system and digital security partners.
(The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)

* “id01 to id11 are numerical features for identity, which is collected by Vesta and security partners such as device rating, ip_domain rating, proxy rating, etc. Also it recorded behavioral fingerprint like account login times/failed to login times, how long an account stayed on the page, etc. All of these are not able to elaborate due to security partner T&C. I hope you could get basic meaning of these features, and by mentioning them as numerical/categorical, you won't deal with them inappropriately.”

**Labeling logic : 
"The logic of our labeling is define reported chargeback on the card as fraud transaction (isFraud=1) and transactions posterior to it with either user account, email address or billing address directly linked to these attributes as fraud too. If none of above is reported and found beyond 120 days, then we define as legit transaction (isFraud=0).
However, in real world fraudulent activity might not be reported, e.g. cardholder was unaware, or forgot to report in time and beyond the claim period, etc. In such cases, supposed fraud might be labeled as legit, but we never could know of them. Thus, we think they're unusual cases and negligible portion."**

**Please note the last paragraph. Most of the participants solved the wrong problem (although they still got good scores !!). Here we do not have to classify whether a transaction is fraudulent or not, because as per the last para all transactions after a fraud transaction for a client is marked fraudulent. So our main motive should be to identify the client for a particular transaction. I know there would still be many questions in your mind, answers of which you would get ahead.**

## <a name="exploration"></a> Exploration

After running some simple exploration scripts, I quickly found out that there are a few aspects of this dataset which makes the competition extremely challenging.

### Class Distribution is Highly Unbalanced

<figure>
  <a href="/images/target.png"><img style="display: block; margin: auto;" src="/images/target.png" alt="Target"></a>
</figure>

### Different distribution of train and test datasets

Here , I would like to mention some important points. Firstly, based on our experiments we found that there were new users in test dataset (i.e. there were users who did not have a single transaction present in train dataset. Secondly, the transaction dates in train and test datasets were disjoint. We had to predict in the future using the past.

<figure>
  <a href="/images/date.png"><img style="display: block; margin: auto;" src="/images/date.png" alt="Transaction Date"></a>
</figure>

## <a name="feature-engineering"></a> Feature Engineering

### Adversial Validation

This is the most important part of this competition. As I have pointed out earlier that the test dataset contained a lot of new users , so **adversial validation** could be used to find out important features to distinguish users. I will not delve deeper into adversial validation , but the main concept of it is that we mark train data as 1 and test data as 0. Then we train a classifier to distinguish between train and test. The most important features used by the model can be used to distinguish users. Below are the most important features learned during adversial validation.

<figure>
  <a href="/images/features.png"><img style="display: block; margin: auto;" src="/images/features.png" alt="Features"></a>
</figure>

### Converting 'days from a certain event' to a point in time

Certian features like D15 were increasing with time. Since they were number of days from a certain event, the increasing relation proved that event was a fixed one. So we can tranform such features to a point in time by subtracting their original value from transaction date. This transformed feature will make life easier for our models.

## <a name="validation-strategy"></a> Validation Strategy

I experimented with a lot of validation strategies. Here are few of them -

### Gradual incrementation

In this we sort the data by time. Then we divide it into say 5 parts. During 1st fold, we train on first part and validate on second. During second fold , we train on first and second, and validate on third. And so on for all folds. This strategy did not provide us good results. One obvious reason was that during the initial folds , we are leaving out almost all data. So during the initial folds we get inaccurate results.

<figure>
  <a href="/images/validation.png"><img style="display: block; margin: auto;" src="/images/validation.png" alt="Validation"></a>
</figure>

### Grouped K-folds

This was the validation strategy we mostly relied on . In this strategy , we grouped on months . Suppose we have months Dec 2012, Jan 2013, Feb 2013 as the only months . So in first fold we trained on Jan and Feb and validated on December. In second fold we trained on Feb and Dec and validated on January and so on . I received a lot of questions as to why I am validating on past using future data. The reason is that distribution of the features are not changing with time. So in a sense there is not any specific trend on time feature. This allowed us to validate effectively using this strategy.

## <a name="models"></a> Models

I used an ensemble of LightGBM , CatBoost and XGBoost. All the models were hypertuned individually using random search cv. We tried several ensembling techniques but we found simple average performs the best. My ensemble scored 0.931068 AUROC on private leaderboard (AUROC is generally the best metric to use when there is huge class imbalance) .

## <a name="conclusion"></a> Conclusion

Although I could not participate in the actual competition , I was able to reach within 1% of Kaggle Leaderboard with the above approach . I have not stated the whole EDA nor the feature engineering part in this blog. Only the important parts are mentioned. To reach within top 1%, a few more bits are needed. I would recommend you to try the above approaches and feel free to contact me for any queries. To end with , I would like to state that you could always use **Label Encoding** to gain better scores.
