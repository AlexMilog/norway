import numpy as np
from matplotlib import pyplot as plt

N1, N2, N3 = 100, 200, 100
L = N1 + N2 + N3
lifetime = 1000
realizations = 10
startpoint = 0
step, endpoint = 0.1, 1.1
ro_range = np.arange(startpoint, endpoint, step)
#c0_range = [5, 10, 20, 50, 100, 150]
c0_range = [5]


def swap(listt, pos1, pos2):
    listt[pos1], listt[pos2] = listt[pos2], listt[pos1]
    return listt


for c0 in c0_range:
    print(c0)
    c = c0 / N2
    output = []
    inputt = []
    for ro in ro_range:
        current_counter = 0
        ro_zero = 0
        print('ro = ', f'{ro:.2f}')
        for realization in range(realizations):
            print('realization = ', realization)
            rna = np.zeros((L,), dtype=np.int)
            for t in range(lifetime):
                filled = []
                for i in range(len(rna)):
                    if rna[i] != 0:
                        filled.append(i)
                if filled:
                    insertion = np.random.choice(np.arange(len(filled)))
                    for i in range(len(filled)):
                        if not filled:
                            continue
                        choice = np.random.choice(filled)
                        e_prob = np.random.uniform(0, 1)
                        if insertion == i:
                            p = np.random.uniform(0, 1)
                            if p <= ro and rna[0] == 0:
                                rna[0] = 1
                        if choice != N1 - 1 and choice != L - 1 and rna[choice] == 1 and rna[choice + 1] == 0:
                            swap(rna, choice, choice + 1)
                            del filled[filled.index(choice)]
                        elif choice != N1 + N2 - 1 and rna[choice] == 2 and rna[choice + 1] == 0:
                            swap(rna, choice, choice + 1)
                            del filled[filled.index(choice)]
                        elif choice == N1 - 1 and e_prob <= c and rna[choice + 1] == 0:
                            swap(rna, choice, choice + 1)
                            rna[choice + 1] = 2
                            del filled[filled.index(choice)]
                        elif choice == N1 - 1 and e_prob > c and rna[choice + 1] == 0:
                            swap(rna, choice, choice + 1)
                            del filled[filled.index(choice)]
                        elif choice != N1 + N2 - 1 and rna[choice] == 2 and rna[choice + 1] == 1:
                            swap(rna, choice, choice + 1)
                            rna[choice] = 0
                            del filled[filled.index(choice)]
                            del filled[filled.index(choice + 1)]
                        elif choice == N1 + N2 - 1 and rna[choice] == 2:
                            rna[choice] = 0
                            del filled[filled.index(choice)]
                        elif choice == L - 1:
                            rna[choice] = 0
                            current_counter += 1
                            del filled[filled.index(choice)]
                else:
                    p = np.random.uniform(0, 1)
                    if p <= ro and rna[0] == 0:
                        rna[0] = 1
                if rna[0] == 1:
                    ro_zero += 1
        ro_zero = ro_zero / (realizations * lifetime)
        inputt.append(ro_zero)
        current_counter = current_counter/(2*realizations * lifetime)
        output.append(current_counter)
    print(inputt)
    print(output)
    plt.plot(inputt, output, 'ro')
    plt.xlim(0, 1)
    plt.show()
