from surprise import SVD, evaluate, accuracy, Reader
import numpy as np
from surprise import Dataset
import pandas as pd
import matplotlib.pyplot as plt



def error(y, model):
	error = 0
	for row in y:
		error += (row[2] - model.predict(str(row[0]), str(row[1])).est)**2

	return (0.5 * error / len(y))


def main():
    train_y = np.loadtxt('data/train.txt').astype(int)
    test_y = np.loadtxt('data/test.txt').astype(int)

    reader = Reader()
    Y_train = Dataset.load_from_file('data/train.txt', reader)
    Y_train  = Y_train.build_full_trainset()


    # regularization factor (0.1 was the best)
    regs = [10**-4, 10**-3, 10**-2, 10**-1, 1]
    # learning rates (0.01 was the best)
    eta = [0.005, 0.01, 0.03, 0.05, 0.07] 
    # number of Latent Factors (5 is the best)
    Ks = [5, 10, 15, 20, 30, 40]
    E_ins = []
    E_outs = []

    # Use to compute Ein and Eout
    for reg in regs:
        E_ins_for_lambda = []
        E_outs_for_lambda = []
        
        for k in Ks:
            
            print('MODEL')
            algo = SVD(n_factors=k, n_epochs=300, biased=True, lr_all = 0.01, reg_all = reg)
            algo.fit(Y_train)
            e_in = error(train_y, algo)
            E_ins_for_lambda.append(e_in)
            eout = error(test_y, algo)
            E_outs_for_lambda.append(eout)
 
        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)

    for i in range(len(regs)):
        plt.plot(Ks, E_ins[i], label='$E_{in}, \lambda=$'+str(regs[i]))
    plt.title('$E_{in}$ vs. Number of Latent Factors (K)')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('E_in_SURPRISE(Latent Factors).png')	
    plt.clf()

    for i in range(len(regs)):
    	plt.plot(Ks, E_outs[i], label='$E_{out}, \lambda=$'+str(regs[i]))
    plt.title('$E_{out}$ vs. Number of Latent Factors (K)')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.legend()	
    plt.savefig('E_out_SURPRISE(Latent Factors).png')	


def bestmodel():
	train_y = np.loadtxt('data/train.txt').astype(int)
	test_y = np.loadtxt('data/test.txt').astype(int)
	reader = Reader()
	Y_train = Dataset.load_from_file('data/train.txt', reader)
	Y_train  = Y_train.build_full_trainset()

	# building the model
	algo = SVD(n_factors=25, n_epochs=300, biased=True, lr_all = 0.01, reg_all = 0.1)
	algo.fit(Y_train)
	# get the errpr
	err = error(test_y, algo)
	print("error:", err)


if __name__ == "__main__":
    #main()
    bestmodel()











