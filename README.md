# Handwritten digit classiﬁcation using multi-class SVM with a Gaussian kernel. 
In order to solve the optimization problem for the SVM, python interface to the LIBSVM package is used. (hhttps://www.csie.ntu.edu.tw/~cjlin/libsvm/)
The training & test data used is USPS dataset. (The dataset refers to numeric data obtained from the scanning of handwritten digits from envelopes by the U.S. Postal Service).

The implementation is done using two different schemes, namely OneVsOne & OneVsRest. The performance of both the schemes is summarized below.

### The test errors for respective cases: 
f1_score SVM One Vs One : 0.928087507652
f1_score SVM One Vs Rest : 0.9303908857

### Wrongly classified digits in both the cases:
![onevsone](https://user-images.githubusercontent.com/15859199/37977561-92291e22-3201-11e8-81a5-cd054d1c9592.png)
![onevsrest](https://user-images.githubusercontent.com/15859199/37977575-9b1f652c-3201-11e8-9597-9551214e0f95.png)
  
### Verdict:
Quality of One Vs Rest > Quality of One Vs One.
The prior results in 129 mismatches and the latter in 134. Thus, there is very slight difference (approximately 1%) in quality of the two classifiers in terms of mismatches but the computation required for One vs One is significantly higher as it runs for nC2 times while One vs Rest does runs only for n times. Visual inspection reveals that both the models wrongly classified almost same instances.
