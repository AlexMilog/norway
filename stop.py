from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import math


def vectorfield0(w, t, p):
    x1, x3, x2, x4 = w
    limit, smallk, bigk, delta = p
    stop = min(limit, max(-limit, x1-x2))
    f = smallk*(x1-x2) + stop
    func = [x3, -f-x2*bigk, x4, f-x4*delta]
    return func


bigk0 = 1
delta0 = 0.01
smallk0 = 30
limit = 1
ss0 = 0

# Initial conditions
x1, x3, x2, x4 = 1, 0.01, 0.01, 0.01

stoptime0 = 40
numpoints0 = 5000

t = [stoptime0 * float(i) / (numpoints0 - 1) for i in range(numpoints0)]
p = [limit, smallk0, bigk0, delta0]
w0 = [x1, x3, x2, x4]
wsol = odeint(vectorfield0, w0, t, args=(p,), atol=1.0e-8, rtol=1.0e-6)

x1_sol = []
x2_sol = []
x3_sol = []
x4_sol = []
for n in range(len(t)):
    x1_sol.append(wsol[n][0])
    x2_sol.append(wsol[n][2])


fig, ax = plt.subplots(1, 2)
plt.subplots_adjust(left=0.15, bottom=0.3)

solx = []
for i in range(0, len(wsol)):
    solx.append(wsol[i][0]-wsol[i][2])


s0 = 0
sols = [s0]
for i in range(0, len(wsol)-1):
    sols.append(min(limit, max(-limit, (((wsol[i+1][0]-wsol[i+1][2])-(wsol[i][0]-wsol[i][2]))+sols[i]))))


l, = ax[0].plot(solx, sols, '-', lw=2)
l1, = ax[0].plot(solx[0], sols[0], 'r*', lw=2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
l2, = ax[1].plot(x1_sol, x2_sol, lw=2)
eq = -np.array(sols)/smallk0
l3, = ax[1].plot(eq, np.zeros(len(eq)), 'r*', lw=2)
ax[0].margins(x=0)
ax[1].margins(x=0)
ax[0].set_xlabel('x1-x2', fontsize=20)
ax[0].set_ylabel('stop', fontsize=20)
ax[1].set_xlabel('x1', fontsize=20)
ax[1].set_ylabel('x2', fontsize=20)
ax[0].set_xlim([-5, 5])
ax[0].set_ylim([-1.1, 1.1])
ax[1].set_xlim([-200, 200])
ax[1].set_ylim([-200, 200])
#plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()


axcolor = 'lightgoldenrodyellow'

smallkval = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
ssmallkval = Slider(smallkval, 'k', 0, 50.0, valinit=smallk0, valstep=0.1)

bigkval = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
sbigkval = Slider(bigkval, 'K', 0, 15.0, valinit=bigk0, valstep=0.1)

sval = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ssval = Slider(sval, 'Initial stop value s0', -1, 1, valinit=ss0, valstep=0.1)


def update(val):
    def vectorfield(w, t, p):
        x1, x3, x2, x4 = w
        limit, smallk, bigk, delta = p
        stop = min(limit, max(-limit, x1-x2))
        f = smallk*(x1-x2)+stop
        func = [x3, -f - x2 * bigk, x4, f - x4 * delta]
        return func

    smallk = ssmallkval.val
    bigk = sbigkval.val
    s = ssval.val
    delta = delta0
    limit = 1

    # Initial conditions
    x1, x3, x2, x4 = 1, 0.01, 0.01, 0.01

    stoptime = stoptime0
    numpoints = numpoints0

    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    p = [limit, smallk, bigk, delta]
    w0 = [x1, x3, x2, x4]
    wsol = odeint(vectorfield, w0, t, args=(p,), atol=1.0e-8, rtol=1.0e-6)

    x1_sol = []
    x2_sol = []
    x3_sol = []
    x4_sol = []
    for n in range(len(t)):
        x1_sol.append(wsol[n][0])
        x2_sol.append(wsol[n][2])
        x3_sol.append(wsol[n][1])
        x4_sol.append(wsol[n][3])

    solx = []
    for i in range(0, len(wsol)):
        solx.append(wsol[i][0] - wsol[i][2])

    s = ssval.val
    sols = [s]
    for i in range(0, len(wsol) - 1):
        sols.append(min(limit, max(-limit, ((wsol[i + 1][0] - wsol[i + 1][2]) - (wsol[i][0] - wsol[i][2]) + sols[i]))))
    eq = -np.array(sols)/smallk

    l.set_xdata(solx)
    l.set_ydata(sols)
    l1.set_xdata(solx[0])
    l1.set_ydata(sols[0])
    l2.set_xdata(x1_sol)
    l2.set_ydata(x2_sol)
    l3.set_xdata(eq)
    fig.canvas.draw_idle()


ssmallkval.on_changed(update)
sbigkval.on_changed(update)
ssval.on_changed(update)

plt.show()
