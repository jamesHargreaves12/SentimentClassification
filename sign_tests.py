import math

def get_significance_ab(a_sent, b_sent, actual_sent):
    negative_count = 0
    for i,v in enumerate(actual_sent):
        if b_sent[i] == v and a_sent[i] != v:
            negative_count += 1
        elif b_sent[i] == a_sent[i]:
            negative_count += 0.5
    print(negative_count)
    prob = 0
    N = len(actual_sent)
    for i in range(math.ceil(negative_count)):
        prob += nCr(N, i) * pow(2,-i)*pow(2,i-N)
    return prob

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

print(get_significance_ab([2,1,2,2,1],[1,2,1,1,2],[1,1,2,2,1]))

# TODO change the way 'main.py' works such that it returns a list of the predictions
#   can add a method to get the % correct
#   then compare systems can change the config file, then run specified fun multiple times with differnt configs to feed into significance test