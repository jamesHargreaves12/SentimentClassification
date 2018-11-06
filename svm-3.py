# SVM

def format_data(data, labels):
    sparse_data = sparse.csr_matrix(data)
    norm_data = normalize(sparse_data).toarray()
    # norm_data = [norm_list(vec) for vec in data]
    # (<label>, [(<feature>, <value>), ...])
    tup_list = []
    for j in range(len(norm_data)):
        vec = norm_data[j]
        feature_tuples = []
        for i in range(len(vec)):
            if vec[i] != 0:
                feature_tuples.append((i, vec[i]))
        tup_list.append((labels[j], feature_tuples))
    return tup_list


def svm_rrscv(rrscv):
    scores = []
    predictions = []
    for i in range(len(rrscv)):
        datas = rrscv[i]
        formatted_data = format_data(datas[0], datas[2])
        svm_model = svm.learn(formatted_data, type='classification')
        fold_predictions = svm.classify(svm_model, format_data(datas[1], [0] * len(datas[3])))

        score = sum([x / abs(x) for x in fold_predictions][j] == datas[3][j] for j in range(len(datas[3])))
        predictions = predictions + fold_predictions
        scores.append(score)
    print("Average: " + str(sum(scores) / len(scores)))
    return predictions


print('SVM')

svm_3_uni = svm_rrscv(rrscv_3)
