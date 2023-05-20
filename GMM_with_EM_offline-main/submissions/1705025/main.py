import sys
from data_loader import DataLoader
from gmm import GMM

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


low = 1
high = 10

def random_color():
    colors=["red","green","blue","orange","purple","pink","brown","gray","olive","cyan"]

    return colors[random.randint(0,len(colors)-1)]

def main():
    dl = DataLoader("./data6D.txt")

    data, dim = dl.getData()

    log_likelihoods = []

    for k in range(low, high+1):
        print("k = ", k)
        gmm = GMM(data, dim, k)
        mu, sigma, pi, log_likelihood = gmm.EM()
        log_likelihoods.append(log_likelihood)

    #--------------------------Task-1--------------------------------
    #plot the log likelihoods vs k
    plt.plot(range(low, high+1), log_likelihoods)
    plt.xlabel("k")
    plt.ylabel("log likelihood")
    plt.show()

    #--------------------------Task-2--------------------------------
    #input the elbow value of k (choose from the plot before manually)
    if dim == 2:
        elbow = int(input("Enter the elbow value of k: "))

        #clear the previous plot
        plt.clf()

        #run the EM algorithm with the elbow value of k
        gmm = GMM(data, dim, elbow)
        gmm.EM()
        means, covs = gmm.get_all_mu_sigma()

        show_animation(data,elbow,means, covs)


def show_animation(data,k,means,covs):

    fig, ax = plt.subplots()

    #get min and max values of x and y from the data
    x_min, x_max = data[:,0].min(), data[:,0].max()
    y_min, y_max = data[:,1].min(), data[:,1].max()

    ax.scatter(data[:,0], data[:,1], s=1)


    #generate random color for each cluster
    colors = []
    for i in range(k):
        colors.append(random_color())


    def update(num):
        ax.cla()

        #plot the data points
        ax.scatter(data[:,0], data[:,1], s=1)
        mean = means[num]
        cov = covs[num]

        for i in range(k):
            col = colors[i]
            plot_contour(ax, mean[i], cov[i], x_min, x_max, y_min, y_max,edgecolor=col, facecolor='none', alpha=0.5 )


        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('iteration: {}'.format(num))
        ax.set_title('Gaussian Mixture Model')


    ani = FuncAnimation(fig, update, frames=range(len(means)), interval=200)

    #save the animation in a gif file
    ani.save('task2.gif', writer='imagemagick', fps=2)
    plt.show()


def plot_contour(ax, mean, cov, x_min, x_max, y_min, y_max, edgecolor='red', facecolor='none', alpha=0.5):
    rv = multivariate_normal(mean, cov)
    x, y = np.mgrid[x_min:x_max:.01, y_min:y_max:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    #make the grids invisible
    ax.contour(x, y, rv.pdf(pos), colors=edgecolor, alpha=alpha)




if __name__ == '__main__':
    main()