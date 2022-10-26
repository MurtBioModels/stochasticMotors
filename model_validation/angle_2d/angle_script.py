import numpy as np
import matplotlib.pyplot as plt

# Parameters
radius = 300
rest_length = 35
km = 0.21
eps0_1 = 0.91
eps0_2 = 7.62
dp_v1 = np.array([2.90, 2.25])
dp_v2 = np.array([0, 0.18])

# Start position and angle
start_angle = np.arcsin(radius/(radius+rest_length))
print(np.rad2deg(start_angle))
start_pos = np.cos(start_angle)*(radius+rest_length)
print(start_pos)
epsilons = []
angles = []
xs = []
f_x = []
f_z = []

for pos in range(150, 198, 8):
    xs.append(pos)
    current_angle = np.arctan(radius/pos)
    angles.append(np.rad2deg(current_angle))
    fx = km * (pos-start_pos)
    #fx = km * pos
    fz = fx * np.tan(current_angle)
    f_x.append(fx)
    f_z.append(fz)

    f_v = np.array([fx, fz])
    k1 = eps0_1 * np.exp((np.dot(f_v,dp_v1)) / 4.1)
    k2 = eps0_2 * np.exp((np.dot(f_v,dp_v2)) / 4.1)
    #print(k1)
    #print(k2)
    eps = k1*k2/(k1+k2)
    epsilons.append(eps)

print(epsilons)
print(angles)
print(xs)
print(f_x)
print(f_z)

plt.plot(f_x, epsilons, label='unbinding rate', linestyle=':', marker='^', color='y')
plt.xlabel('Horizontal force Fx [pN]')
plt.ylabel('Unbinding rate [1/s]')
plt.grid()
plt.title(f'Changing angle: f_x vs detachment rate')
plt.legend()
plt.show()
'''
dp_v1 = np.array([2.90, 2.25])
dp_v2 = np.array([0, 0.18])
eps0_1 = 0.91
eps0_2 = 7.62
Boltzmann = 1.38064852e-23
T = 4.1/Boltzmann
angle = 60 * (np.pi / 180)

un_z_only = []
fz_list = []
for fz in range(0,20,1):

    fx = 0
    f_v = np.array([fx, fz])

    k1 = eps0_1 * np.exp((np.dot(f_v,dp_v1)) / 4.1)
    k2 = eps0_2 * np.exp((np.dot(f_v,dp_v2)) / 4.1)

    un_z_only.append(k1*k2/(k1+k2))
    fz_list.append(fz)


un_x_only = []
fx_only_list = []
for fx in range(-20,20,1):

    fz = 0
    f_v = np.array([fx, fz])

    k1 = eps0_1 * np.exp((np.dot(f_v,dp_v1)) / 4.1)
    k2 = eps0_2 * np.exp((np.dot(f_v,dp_v2)) / 4.1)

    un_x_only.append(k1*k2/(k1+k2))
    fx_only_list.append(fx)

unbinding_tot = []
fx_tot_list = []
fz_component = []
for fx in range(-20,20,1):

    fz = abs(fx)*np.tan(angle)
    fz_component.append(fz)

    f_v = np.array([fx, fz])

    k1 = eps0_1 * np.exp((np.dot(f_v,dp_v1)) / 4.1)
    k2 = eps0_2 * np.exp((np.dot(f_v,dp_v2)) / 4.1)

    unbinding_tot.append(k1*k2/(k1+k2))
    fx_tot_list.append(fx)

fig, ax1 = plt.subplots()
ax1.plot(fx_only_list, un_x_only, label='fx only', linestyle=':', marker='x', color='g')
ax1.plot(fx_tot_list, unbinding_tot, label='fx (when fz is also acting)', linestyle=':', marker='^', color='y')
ax1.spines['left'].set_position(('data', 0))
ax1.set_xlabel('Horizontal force Fx [pN]')

ax2 = ax1.twiny()
ax2.plot(fz_list, un_z_only, label='fz only', linestyle=':', marker='o', color='r')
ax2.set_xlim(-20, 20)
ax2.set_xlabel('Vertical force Fz [pN]')

plt.ylabel('unbinding rate (1/s)')
plt.yticks(np.arange(0, max(unbinding_tot), step=5))
fig.suptitle(f'Force detachment rate')
ax1.legend()
ax2.legend()
ax1.grid()
plt.show()
'''



