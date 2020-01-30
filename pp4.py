import random
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time
from numpy import linalg as L2
import collections

def read_data(file):
    data_corpus = []
    for i in range(200):
            f = open(file + str(i + 1))
            for row in f:
                data_corpus.append(row.strip().split(' '))
            f.close()


    return data_corpus
file="./20newsgroups/"
data_corpus=read_data(file)

def requirement( K, data_corpus):

    no_of_documents=len(data_corpus)
    alpha= 5/K
    beta=0.01

    A=[]
    total_words_N_words=[]
    for i in range(no_of_documents):
        for j in range(len(data_corpus[i])):
            total_words_N_words.append(data_corpus[i][j])

            if data_corpus[i][j] not in A:
                A.append(data_corpus[i][j])


    document_indices=[]
    z_topic_indices=[]
    w_indices=[]
    for i in range(no_of_documents):
        for j in range(len(data_corpus[i])):
            w_indices.append(A.index(data_corpus[i][j]))

            document_indices.append(i)
            z_topic_indices.append(random.randint(0, K-1))
    w_indices=np.array(w_indices)
    document_indices=np.array(document_indices)

    z_topic_indices=np.array(z_topic_indices)
    return w_indices,document_indices,z_topic_indices,A, total_words_N_words,K

w_indices,document_indices,z_topic_indices,A,total_words_N_words,K=requirement(20, data_corpus)

permutation_pi_n=np.random.permutation(range(len(total_words_N_words)))

c_d_counts_per_document =  np.zeros((len(data_corpus),K))
c_t_counts_per_topic=np.zeros((K,len(A)))
P=np.zeros(K)
for i in range(len(total_words_N_words)):
    c_d_counts_per_document[document_indices[i]][z_topic_indices[i]]+=1
    c_t_counts_per_topic[z_topic_indices[i]][w_indices[i]]+=1
def gibbs_sampling(N_iters,permutation_pi_n,c_d_counts_per_document,c_t_counts_per_topic,K,w_indices,document_indices,z_topic_indices,A,P,total_words_N_words):
    beta=0.01
    alpha=5.0/K
    random.seed(1)

    for i in range(N_iters):
        for n in range(0,len(total_words_N_words)):
            word = w_indices[permutation_pi_n[n]]
            topic= z_topic_indices[permutation_pi_n[n]]
            doc  = document_indices[permutation_pi_n[n]]
            c_d_counts_per_document[doc][topic]=c_d_counts_per_document[doc][topic]-1
            c_t_counts_per_topic[topic][word]=c_t_counts_per_topic[topic][word]-1

            for k in range(K):
                P[k]=((c_t_counts_per_topic[k][word]+beta)/(len(A)*beta + np.sum(c_t_counts_per_topic[k, : ])))*((c_d_counts_per_document[doc][k]+alpha)/(K * alpha + np.sum(c_d_counts_per_document[doc, : ])))
            P=np.divide(P,np.sum(P))
            rang=[m for m in range(K)]
            topic = np.random.choice(rang,p=P)
            z_topic_indices[permutation_pi_n[n]]=topic
            c_d_counts_per_document[doc][topic]=c_d_counts_per_document[doc][topic]+1
            c_t_counts_per_topic[topic][word]=c_t_counts_per_topic[topic][word]+1

    return z_topic_indices, c_d_counts_per_document,c_t_counts_per_topic


z_topic_indices, c_d_counts_per_document,c_t_counts_per_topic=gibbs_sampling(500,permutation_pi_n,c_d_counts_per_document,c_t_counts_per_topic,20,w_indices,document_indices,z_topic_indices,A,P,total_words_N_words)
freq_words_index=[]

for i in range(K):

    sorted_row=c_t_counts_per_topic[i]

    index=np.argsort(sorted_row)

    freq_words_index.append(index[-5:])


topic_words=[]
for i in range(len(freq_words_index)):
    row_i=[]
    for ele in freq_words_index[i]:
        row_i.append(A[ele])
    topic_words.append(row_i)

with open("topicwords.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(topic_words)

def vector_of_k( c_d_counts_per_document,K,data_corpus):
    alpha=5/K
    k_vector=[]
    for i in range(len(data_corpus)):
        temp=[]
        for k in range(K):
            temp.append((c_d_counts_per_document[i][k]+alpha)/(K * alpha + np.sum(c_d_counts_per_document[i, : ])))

        k_vector.append(temp)
    k_vector=np.array(k_vector)
    return k_vector
data_vector= vector_of_k( c_d_counts_per_document,K,data_corpus)

k_data= pd.DataFrame(data_vector)

total_words_dict=collections.Counter(total_words_N_words)
l_words=list(total_words_dict.keys())
bow_matrix = np.zeros((len(data_corpus),len(A)))
for i in range(len(data_corpus)):
    for j in data_corpus[i]:
        if j in l_words:
            ind = l_words.index(j)
            bow_matrix[i][ind]=1

bow_m_data= pd.DataFrame(bow_matrix)


def read_stuff(label):
    labels_df = pd.read_csv(label,header=None)

    return labels_df
label="/Users/kartik/PycharmProjects/ML Assignment 4/20newsgroups/index.csv"
labels_df=read_stuff(label)
labels_df.drop(labels_df.columns[[0]], axis=1, inplace=True)

def pd_to_numpy_converter(k_data,labels_df):
    data_vector=k_data.values
    label_vector= labels_df.values

    return data_vector,label_vector

def Id_matrix_generator(data_vector):
    Identity_matrix = np.identity(data_vector.shape[1])

    return Identity_matrix

def generate_w(data_vector):
    w=[0]*data_vector.shape[1]
    w=np.array(w)
    w=w.reshape(w.shape[0],1)

    return w

def predictor(data_vector,w):

    y=(1 / (1 + np.exp(-np.dot(data_vector,w))))

    y=y.reshape(data_vector.shape[0],1)

    return y
def predictor_poisson(data_vector,w):

    pois_pred= np.exp(np.dot(data_vector,w))
    pois_pred = pois_pred.reshape(data_vector.shape[0], 1)

    return pois_pred

def predictor_poisson_test(data_vector,w):

    pois_pred= np.floor(np.exp(np.dot(data_vector,w)))
    pois_pred = pois_pred.reshape(data_vector.shape[0], 1)

    return pois_pred


def predict_ordinal_test(w_new, data_vector, s):
    y0 = 0
    y1 = 1 / (1 + np.exp((-s)*(-2 - np.dot(data_vector, w_new))))
    y2 = 1 / (1 + np.exp((-s)*(-1 - np.dot(data_vector, w_new))))
    y3 = 1 / (1 + np.exp((-s)*(0 - np.dot(data_vector, w_new))))
    y4 = 1 / (1 + np.exp((-s)*(1 - np.dot(data_vector, w_new))))
    y5 = 1

    predictor_ordinal = np.array([np.transpose(y1 - y0), np.transpose(y2 - y1), np.transpose(y3 - y2), np.transpose(y4 - y3),
                      np.transpose(y5 - y4)])
    ordinal_predict=np.argmax(predictor_ordinal[:, 0, :], axis=0) + 1
    return ordinal_predict

def y_value(data_vector, label_vector, w_new, s):
    def phi(x_i):
        if x_i == 0:
            return -100
        elif x_i == 1:
            return -2
        elif x_i == 2:
            return -1
        elif x_i == 3:
            return 0
        elif x_i == 4:
            return 1
        elif x_i == 5:
            return 100

    phi = np.array([phi(x_i[0]) for x_i in label_vector])
    phi = np.transpose(phi)
    phi = phi.reshape(-1, 1)
    a_i = np.dot(data_vector, w_new)
    sigmoid_y= 1 / (1 + np.exp(-(s * (np.subtract(phi, a_i)))))
    return sigmoid_y


def y_1_value(data_vector, label_vector, w_new, s):
    def phi(x_i):
        if x_i == 1:
            return -100
        elif x_i == 2:
            return -2
        elif x_i == 3:
            return -1
        elif x_i == 4:
            return 0
        elif x_i == 5:
            return 1

    phi = np.array([phi(x_i[0]) for x_i in label_vector])
    phi = np.transpose(phi)
    phi = phi.reshape(-1, 1)
    a_i = np.dot(data_vector, w_new)
    sigmoid_y_1= 1 / (1 + np.exp(-(s * (np.subtract(phi, a_i)))))
    return sigmoid_y_1


def R_ordinal(data_vector, label_vector, w_new, s):
    y_i_j = y_value(data_vector, label_vector, w_new, s)
    y_i_j_1 = y_1_value(data_vector, label_vector, w_new, s)

    r = (np.multiply(y_i_j, 1 - y_i_j) + np.multiply(y_i_j_1, 1 - y_i_j_1))
    R_ord=np.diag(r.reshape(1, -1)[0])
    return R_ord

def ordinal_d_i(data_vector, label_vector, w_new, s):
    y_i_j = y_value(data_vector, label_vector, w_new, s)
    y_i_j_1 = y_1_value(data_vector, label_vector, w_new, s)
    d_i=y_i_j + y_i_j_1 - 1
    return d_i

def first_derivative(data_vector,label_vector,y,w,alpha=0.01):
    d = np.subtract(label_vector,y)
    first_d=np.subtract(np.dot(np.transpose(data_vector),d),np.multiply(alpha,w))
    return first_d
def first_derivative_ordinal(data_vector,label_vector,d,w,alpha=0.01):
    first_d=np.subtract(np.dot(np.transpose(data_vector),d),np.multiply(alpha,w))
    return first_d
def R_matrix_logistic(y):
    temp=np.multiply(y,1-y)
    temp=temp.reshape(1,-1)[0]
    R = np.diag(temp)
    return R

def R_matrix_poisson(pois_pred):
    temp = pois_pred.reshape(1, -1)[0]
    R_poisson = np.diag(temp)

    return R_poisson

def hessian_matrix(data_vector,R,Identity_matrix,alpha=0.01):
    initial= np.transpose(data_vector).dot(R).dot(data_vector)
    hessian = np.add(initial,np.multiply(alpha,Identity_matrix))
    hessian_inv =np.linalg.inv(hessian)

    return hessian_inv

def newton_update(data_vector,label_vector,regression,alpha):

    count = []
    w = generate_w(data_vector)

    count += [w]
    time_now=time.time()
    for i in range(100):
        if regression=="logistic" :
            y = predictor(data_vector, w)
            Identity_matrix = Id_matrix_generator(data_vector)
            first_d = first_derivative(data_vector, label_vector, y, w, alpha)
            R = R_matrix_logistic(y)
            hessian_inv = hessian_matrix(data_vector, R, Identity_matrix, alpha)
            w=np.add(w,np.dot(hessian_inv,first_d))
        elif regression=="poisson":
            pois_pred=predictor_poisson(data_vector, w)
            Identity_matrix = Id_matrix_generator(data_vector)
            first_d = first_derivative(data_vector, label_vector, pois_pred, w, alpha)
            R_poisson=R_matrix_poisson(pois_pred)
            hessian_inv = hessian_matrix(data_vector, R_poisson, Identity_matrix, alpha)
            w = np.add(w, np.dot(hessian_inv, first_d))
        elif regression == "ordinal":
            d = ordinal_d_i(data_vector, label_vector, w, 1)
            Identity_matrix = Id_matrix_generator(data_vector)
            first_d = first_derivative_ordinal(data_vector,label_vector,d,w,alpha)
            R_ord = R_ordinal(data_vector, label_vector, w, 1)
            hessian_inv = hessian_matrix(data_vector, R_ord,Identity_matrix,alpha)
            w = np.add(w, np.dot(hessian_inv, first_d))

        count = count+[w]
        if (L2.norm(count[i]!=0)):
            if (L2.norm(count[i+1]-count[i])/L2.norm(count[i])) < 0.001:
                return w,i,time.time()-time_now

    return w,i,time.time()-time_now
#data_vector,label_vector=pd_to_numpy_converter(data_table_w,labels_df)
#newton_update(data_vector,label_vector,regression='ordinal',alpha=10)

def sample_split(data_table_w,labels_df):
    data_table_w_copy=data_table_w.copy()
    label_df_copy=labels_df.copy()
    train_data=data_table_w_copy.sample(frac=2/3, random_state=15)
    train_label=label_df_copy.sample(frac=2/3, random_state=15)

    test_data=data_table_w_copy.drop(train_data.index)
    test_label=label_df_copy.drop(train_label.index)

    return train_data,train_label,test_data,test_label
def split_data(data,labels):
    sampled_data=data.sample(frac=0.33)
    indice=sampled_data.index
    sampled_labels=labels.loc[indice,:]
    train_data=data.loc[~data.index.isin(indice)]
    train_labels=labels.loc[~labels.index.isin(indice)]
    return train_data,train_labels,sampled_data,sampled_labels
def predict_classifier(test_data_vector,w_new):

    predicted_test_label_vector=predictor(test_data_vector, w_new)
    for element in range (predicted_test_label_vector.shape[0]):
        if predicted_test_label_vector[element] >= 0.5:
            predicted_test_label_vector[element]=1
        else:
            predicted_test_label_vector[element]=0

    return predicted_test_label_vector


def errors(predicted_test_label_vector,test_label_vector):

    error =np.mean(np.absolute((predicted_test_label_vector-test_label_vector)))

    return error

def poisson_error(pois_pred,test_label_vector):
    poiss_error= np.sum(np.absolute(test_label_vector-pois_pred))

    return poiss_error

def proportion(proportion,train_data,train_label):
    train_data_copy = train_data.copy()
    train_label_copy = train_label.copy()
    proportion_train_data = train_data_copy.sample(frac=proportion,random_state=15)
    proportion_train_label = train_label_copy.sample(frac=proportion,random_state=15)

    return proportion_train_data,proportion_train_label
def divides(fraction,train_data,train_labels):
    sampled_data=train_data.sample(frac=fraction)
    indice=sampled_data.index
    sampled_labels=train_labels.loc[indice,:]
    return sampled_data,sampled_labels
def final_main(data_table_w, labels_df):
    time_right_now = time.time()
    error_array=[[] for k in range (0,10)]
    ts = [[] for k in range(0, 10)]
    ites = [[] for k in range(0, 10)]
    for element in range(0,30):

        train_data,train_label,test_data_2,test_label=split_data(data_table_w, labels_df)

        for i in range(1,11):
            proportion_train_data,proportion_train_label=divides(i/10,train_data,train_label)
            train_data_vector, train_label_vector = pd_to_numpy_converter(proportion_train_data, proportion_train_label)

            test_data_vector_2,test_label_vector = pd_to_numpy_converter(test_data_2, test_label)

            w_new,ite,t = newton_update(train_data_vector, train_label_vector, regression='logistic',alpha=0.01)

            predicted_test_label_vector = predict_classifier(test_data_vector_2, w_new)

            error= errors(predicted_test_label_vector, test_label_vector)

            error_array[i-1]+=[error]
            ites[i - 1] += [ite]
            ts[i - 1] += [t]
    err=[]
    sds=[]
    itts=[]
    tss=[]
    print("Total time", time.time() - time_right_now)
    for k in range(0,10):
        err+=[1-np.mean(error_array[k])]
        tss += [np.mean(ts[k])]
        itts += [np.mean(ites[k])]
        sds+=[np.std(error_array[k])]


    return err,sds
def final_main_poisson(data_table_w, labels_df):
    time_right_now = time.time()
    error_array=[[] for k in range (0,10)]
    ts = [[] for k in range(0, 10)]
    ites = [[] for k in range(0, 10)]
    for element in range(0,30):
        train_data, train_label, test_data, test_label = split_data(data_table_w, labels_df)
        for i in range(1, 11):
            proportion_train_data, proportion_train_label = divides(i / 10, train_data, train_label)
            train_data_vector, train_label_vector = pd_to_numpy_converter(proportion_train_data, proportion_train_label)
            test_data_vector,test_label_vector = pd_to_numpy_converter(test_data, test_label)
            w_new,ite,t = newton_update(train_data_vector, train_label_vector, regression='poisson',alpha=10)

            pois_pred_test_label_vector=predictor_poisson_test(test_data_vector, w_new)

            error = poisson_error(pois_pred_test_label_vector,test_label_vector)
            error_array[i-1]+=[error/len(pois_pred_test_label_vector)]
            ites[i - 1] += [ite]
            ts[i - 1] += [t]
    final_errors=[]
    final_std=[]
    itts = []
    tss = []
    print("Total time", time.time() - time_right_now)
    for i in range(0,10):
         #print(np.mean(error[i]))
         tss += [np.mean(ts[i])]
         itts += [np.mean(ites[i])]
         final_errors.append(np.mean(error_array[i]))
         final_std.append(np.std(error_array[i]))

    plt.title("Mean absolute error vs Portion of Data")
    plt.xlabel("Portion of train data")
    plt.ylabel("Mean absolute error")
    print(itts, tss)
    plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], final_errors, color='grey')
    plt.errorbar([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], final_errors, final_std)
    plt.show()
def final_main_ordinal(data_table_w, labels_df):
    time_right_now = time.time()
    error_array = [[] for k in range(0, 10)]
    ts = [[] for k in range(0, 10)]
    ites = [[] for k in range(0, 10)]
    for element in range(0, 30):

        train_data, train_label, test_data, test_label = split_data(data_table_w, labels_df)

        for i in range(1, 11):
            proportion_train_data, proportion_train_label = divides(i / 10, train_data, train_label)
            train_data_vector, train_label_vector = pd_to_numpy_converter(proportion_train_data, proportion_train_label)

            test_data_vector, test_label_vector = pd_to_numpy_converter(test_data, test_label)
            w_new, ite, t = newton_update(train_data_vector, train_label_vector, regression='ordinal', alpha=10)

            ordinal_predict=predict_ordinal_test(w_new, test_data_vector, 1)


            error = poisson_error(ordinal_predict.T.reshape(1,-1), test_label_vector.T)


            error_array[i-1]+=[error/len(ordinal_predict)]
            ites[i - 1] += [ite]
            ts[i - 1] += [t]


    final_errors=[]
    final_std=[]
    itts = []
    tss = []
    print("Total time", time.time() - time_right_now)
    for i in range(0,10):
         #print(np.mean(error[i]))
         tss += [np.mean(ts[i])]
         itts += [np.mean(ites[i])]
         final_errors.append(np.mean(error_array[i]))
         final_std.append(np.std(error_array[i]))
    print(final_std[0])
    plt.title("Mean absolute error vs Portion of Data")
    plt.xlabel("Portion of train data")
    plt.ylabel("Mean absolute error")
    print(itts, tss)
    plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], final_errors, color='grey')
    plt.errorbar([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], final_errors, yerr=final_std)
    plt.show()

def main(k_data,bow_m_data):

    err,sds= final_main(k_data, labels_df)
    err_1,sds_1=final_main(bow_m_data, labels_df)

    plt.title("accuracy  vs Portion of Data")
    plt.xlabel("Portion of train data")
    plt.ylabel("accuracy")

    plt.errorbar([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], err, yerr=sds,color='red',capsize=20,label="LDA")
    plt.errorbar([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], err_1, yerr=sds_1,color='blue',capsize=20,label="Bag of words")
    plt.legend()
    plt.show()

main(k_data,bow_m_data)












