import numpy as np
import matplotlib.pyplot as plt

f = np.arange(-8, 8.1, 0.01)

epsilon_0 = 0.66
f_d = 2.1
epsilon = epsilon_0 * (np.e**(abs(f)/f_d))

def step(f_s, alfa_0, f_current):

    alfa_list = []
    for f in f_current:
        if f < 0:
            alfa = alfa_0
        elif f > f_s:
            alfa = 0
        else:
            alfa = alfa_0 * (1 - (f / f_s))
        alfa_list.append(alfa)

    return alfa_list

plt.figure()
plt.plot(f, step(f_current=f, f_s=7, alfa_0=92.5), color='black')
plt.title('Force dependent stepping rate', fontsize=18)
plt.xlabel('Force', fontsize=18)
plt.ylabel('Stepping rate', fontsize=18)
plt.xticks([0], fontsize=18)
plt.yticks([0], fontsize=18)
plt.ylim(ymin=0)
plt.xlim(xmin=-8)
plt.xlim(xmax=8)
plt.axvline(x=0, color='gray', ls=':')
plt.text(7, -5, 'Fs', fontsize='x-large')
plt.text(-9, 92.5, '\u03C30', fontsize='x-large')
#plt.tight_layout()
plt.savefig('figure_1.png', format='png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(f, epsilon, color='black')
plt.title('Force dependent unbinding rate', fontsize=18)
plt.xlabel('Force', fontsize=18)
plt.ylabel('Unbinding rate', fontsize=18)
plt.xticks([0], fontsize=18)
plt.axvline(x=0, color='gray', ls=':')
plt.vlines(x=[-2.1, 2.1], ymin=[0, 0], ymax=[1.79, 1.79], lw=2, colors='gray', ls=':')
plt.text(f_d, -2, 'Fd', fontsize='x-large')
plt.text(-1*f_d, -2, 'Fd', fontsize='x-large')
plt.yticks([0], fontsize=18)
plt.ylim(ymin=0)
plt.xlim(xmin=-8)
plt.xlim(xmax=8)
plt.hlines(y=0.66, xmin=-8, xmax=0, color='black', linestyle=':')
plt.text(-9.5, 0.66, '\u03B50', fontsize='x-large')
plt.savefig('figure_2.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

