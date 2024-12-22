import pickle

import numpy as np
import scipy

with open("metrics.pkl", "rb") as fb:
    data = pickle.load(fb)

print("dt orig", np.sum(data["dt_orig"]) / data["run_flag"])
print("dt tae", np.mean(data["dt_tae"]))
print("dt tae temp", np.mean(data["dt_taetemp"]))


cov_gt = data["cov_gt"] / data["run_flag"]
mean_gt = data["mean_gt"] / data["run_flag"]

cov_orig = data["cov_orig"] / data["run_flag"]
mean_orig = data["mean_orig"] / data["run_flag"]

cov_tae = data["cov_tae"] / data["run_flag"]
mean_tae = data["mean_tae"] / data["run_flag"]

cov_taetemp = data["cov_taetemp"] / data["run_flag"]
mean_taetemp = data["mean_taetemp"] / data["run_flag"]

#
m = np.square(mean_gt - mean_orig).sum()
s, _ = scipy.linalg.sqrtm(np.dot(cov_gt, cov_orig), disp=False)
print("GT-ORIG", np.real(m + np.trace(cov_gt + cov_orig - s * 2)))


m = np.square(mean_gt - mean_tae).sum()
s, _ = scipy.linalg.sqrtm(np.dot(cov_gt, cov_tae), disp=False)
print("GT-TAE", np.real(m + np.trace(cov_gt + cov_tae - s * 2)))


m = np.square(mean_gt - mean_taetemp).sum()
s, _ = scipy.linalg.sqrtm(np.dot(cov_gt, cov_taetemp), disp=False)
print("GT-TAE TEMP", np.real(m + np.trace(cov_gt + cov_taetemp - s * 2)))
