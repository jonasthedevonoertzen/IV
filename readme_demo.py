from sklearn.datasets import load_wine
import independent_validation as iv
from sklearn.svm import SVC

wine = load_wine()
X, y = wine.data, wine.target

iv_svm = iv.IV(X, y, SVC(gamma='scale'))  # initiating
iv_svm.run_iv()  # classify samples
iv_svm.compute_posterior()  # use mcmc to compute posterior
bacc_svm_dist = iv_svm.get_bacc_dist()  # get desired output
print("Mode (MAP) value:", bacc_svm_dist.map())