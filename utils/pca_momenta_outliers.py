import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

momenta = np.loadtxt("../data_models_saved/data/DeterministicAtlas__EstimatedParameters__Momenta.txt")

momenta = np.delete(momenta, 0, axis=0)
momenta = momenta.reshape((2274, 720, 3))

momenta = np.delete(momenta, [1746, 1831], axis=0)

momenta_pca = momenta.reshape(-1, 720 * 3)


pca = PCA(n_components=0.99)
scores = pca.fit_transform(momenta_pca)


mean_scores = np.mean(scores, axis=0)
cov_scores = np.cov(scores, rowvar=False)
inv_cov_scores = np.linalg.inv(cov_scores)  ### Mahalanobis distance using a covariance estimate

### Outlier detection
### based on the fact that the squared Mahalanobis distance approximately follows a chi-square distribution
### with degrees of freedom equal to the PCA dimension.

md = np.array([
    mahalanobis(s, mean_scores, inv_cov_scores)
    for s in scores
])

k = scores.shape[1]   ### number of retained PCs
threshold = np.sqrt(chi2.ppf(0.9999999999999, df=k))

### threshold

outliers = np.where(md > threshold)[0]
print(f'Outliers num: {len(outliers)}, {outliers}')
np.save('../utils/outliers_indices.npy', outliers)

### first two PCs
plt.figure(figsize=(7, 6))
plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7, label="Samples")
plt.scatter(scores[outliers, 0], scores[outliers, 1], s=50, label="Outliers")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title("PCA scores with outliers")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(md, 'o', markersize=4)
plt.axhline(threshold)
plt.xlabel("Sample index")
plt.ylabel("Mahalanobis distance")
plt.title("Outlier detection in PCA space")
plt.show()




