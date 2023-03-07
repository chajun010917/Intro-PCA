from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x,axis=0)
    return x

def get_covariance(dataset):
    dataset = np.dot(np.transpose(dataset),dataset) / (len(dataset)-1)
    return dataset

def get_eig(S, m):
    w,v = eigh(S)
    w1 = np.sort(w)[::-1]
    w1 = np.diag(w1[:m])
    v = v[:, ::-1]
    v = v[:, :m]
    return w1,v

def get_eig_prop(S, prop):
    sum = 0.0
    w,v = eigh(S)
    for x in w:
        sum+=x
    a = [prop*sum, np.inf]
    w,v = eigh(S, subset_by_value=a)
    w = np.sort(w)[::-1]
    w = np.diag(w[:])
    v = v[:, ::-1]
    return w, v

def project_image(image, U):
    projection = np.dot(U, np.dot(np.transpose(U),image))
    return projection

def display_image(orig, proj):
    orig = np.reshape(orig,(32,32))
    orig = orig.transpose()
    proj = np.reshape(proj,(32,32))
    proj = proj.transpose()
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Original")
    ax2.set_title("Projection")
    p1 = ax1.imshow(orig)
    p2 = ax2.imshow(proj)
    fig.colorbar(p1, ax=ax1)
    fig.colorbar(p2, ax=ax2)
    plt.show()
    pass


