#Importing Numpy, Matplotlib, Math, and Pandas
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd 

figure_counter = 0

def extract_data(extension):
    raw_data = pd.read_csv('Sound Lab Data/WED1/' + extension + '.csv', sep = ';', header = None)
    edited_data = raw_data.to_numpy()
    edited_data = np.delete(edited_data, 0, 0)
    edited_data = np.delete(edited_data, 3, 1)
    edited_data = np.delete(edited_data, 0, 1)
    edited_data = edited_data.astype(np.float)
    return edited_data

def return_vivt(edited_data):
    v1 = np.max(edited_data[:, 0])
    v2 = np.max(edited_data[:, 1])
    vi = max(v1, v2)
    vt = min(v1, v2)
    return vi, vt

def return_bf(extension):
    edited_data = extract_data(extension)
    vi, vt = return_vivt(edited_data)
    b_f = math.log10(vi/vt)
    return vi, vt, b_f

def return_tf(b_f, extension):
    edited_data = extract_data(extension)
    vi, vt = return_vivt(edited_data)
    t_f = 20.0*math.log10(vi/vt-b_f)
    return vi, vt, t_f

def plot_normalization(fig_number, freq, val):
    fig = plt.figure(fig_number)
    freq = freq.astype(np.float)
    plt.plot(freq, val)

    #Plot Lengends, Titles, and Saved Images
    plt.legend(['Air'])
    plt.xlabel('Frequency')
    plt.ylabel('B(f)')
    plt.savefig('Normalization_Plot.png')
    fig.show()

def plot_transmission(fig_number, freq, val, mat):
    fig = plt.figure(fig_number)
    freq = freq.astype(np.float)

    for j in range(np.size(mat)):
        plt.semilogy(freq, val[:, j])

    #Plot Lengends, Titles, and Saved Images
    plt.legend([mat[0], mat[1], mat[2], mat[3], mat[4], mat[5]])
    plt.xlabel('Frequency')
    plt.ylabel('TL(f)')
    plt.savefig('Transmission_Loss.png')
    fig.show()

def plot_distance(fig_number, freq, val, dist):
    fig = plt.figure(fig_number)
    dist = dist.astype(np.float)

    for j in range(np.size(freq)):
        plt.semilogy(dist, val[j, :])

    #Plot Lengends, Titles, and Saved Images
    plt.legend([freq[0] + ' Hz', freq[1] + ' Hz', freq[2] + ' Hz'])
    plt.xlabel('Distance (cm)')
    plt.ylabel('Vt')
    plt.savefig('Distance.png')
    fig.show()

#Part 1#############################################
def part1():
    print('##################### ( PART 1 ) #####################')
    global figure_counter
    freq = np.array(['300', '700', '1100', '1500', '1900', '2300', '2700', '3000'])
    mat = np.array(['Ceiling', 'Hollow Drywall', 'Insulated Drywall', 'Steel', 'Steel and Foam', 'Wood'])
    f, m = np.size(freq), np.size(mat)

    vi_air, vt_air, bf_air = np.zeros(f), np.zeros(f), np.zeros(f)
    vi_stored, vt_stored, tf_stored = np.zeros((f, m)), np.zeros((f, m)), np.zeros((f, m))

    print('------------Air------------')
    for i in range(f):
        vi_air[i], vt_air[i], bf_air[i] = return_bf('Air/' + freq[i])
        print(freq[i] + ' & ' + str(vi_air[i]) + ' & ' + str(vt_air[i]) + ' & ' + str(bf_air[i]) + ' \\\\')

    for j in range(m):
        print('------------' + mat[j] + '------------')
        for i in range(f):
            vi_stored[i, j], vt_stored[i, j], tf_stored[i, j] = return_tf(bf_air[i], mat[j] + '/' + freq[i])
            print(freq[i] + ' & ' + str(vi_stored[i, j]) + ' & ' + str(vt_stored[i, j]) + ' & ' + str(tf_stored[i, j]) + ' \\\\')
    
    figure_counter += 1
    plot_normalization(figure_counter, freq, bf_air)
    figure_counter += 1
    plot_transmission(figure_counter, freq, tf_stored, mat)
####################################################

#Part 2#############################################
def part2():
    print('##################### ( PART 2 ) #####################')
    global figure_counter
    freq = np.array(['300', '1650', '3300'])
    dist = np.array(['40', '48', '56', '64', '72', '80', '88', '96', '104', '112', '120', '128', '136', '144', '152', '160', '168', '176'])
    f, d = np.size(freq), np.size(dist)

    vi_stored, vt_stored = np.zeros((f, d)), np.zeros((f, d))

    for i in range(f):
        print('------------' + freq[i] + '------------')
        for j in range(d):
            edited_data = extract_data('Part 2/' + freq[i] + ' - ' + dist[j] + 'cm')
            vi_stored[i, j], vt_stored[i, j] = return_vivt(edited_data)
            print(dist[j] + ' & ' + str(vi_stored[i, j]) + ' & ' + str(vt_stored[i, j]) + ' \\\\')

    figure_counter += 1

    plot_distance(figure_counter, freq, vt_stored, dist)
####################################################

#Part 3#############################################
def part3():
    print('##################### ( PART 3 ) #####################')
    global figure_counter
    timeA = np.array([30, 90, 150, 210, 270, 330, 390, 450, 510])
    timeC = np.array([60, 120, 180, 240, 300, 360, 420, 480])

    dbA = np.array([49.4, 51.6, 51.3, 48.1, 49.9, 47.9, 47.7, 49.0, 50.3])
    dbC = np.array([66.1, 66.4, 66.0, 65.7, 66.2, 64.1, 64.2, 65.5])

    figure_counter += 1

    fig = plt.figure(figure_counter)
    plt.plot(timeA, dbA)
    plt.plot(timeC, dbC)

    print('------------A/C------------')
    for i in range(np.size(timeA)):
        print(str(timeA[i]) + ' & ' + str(dbA[i]) + ' & ' + '-' + ' \\\\')
        if i < np.size(timeC):
            print(str(timeC[i]) + ' & ' + '-' + ' & ' + str(dbC[i]) + ' \\\\')

    #Plot Lengends, Titles, and Saved Images
    plt.legend(['Plot A', 'Plot C'])
    plt.xlabel('Time (s)')
    plt.ylabel('dB')
    plt.savefig('Sound_Level.png')
    fig.show()
####################################################
#Code Execution

part1()
part2()
part3()
input()
####################################################
