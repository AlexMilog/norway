import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import FuncFormatter
import numpy as np


def my_formatter(x, pos):
    if x.is_integer():
        return str(int(x))
    else:
        return str(round(x, 3))


formatter = FuncFormatter(my_formatter)

wpred = np.array([0., 0., 0., 0.])
spred = 0
arrayofs = [spred]


def vectorfield(w, t, p):
    global wpred
    global spred
    global arrayofs

    x1, x2, x3, x4 = w
    smallk, bigk, delta = p
    if t == 0:
        stop = spred
    else:
        stop = min(1.0, max(-1.0, (w[0] - w[1]) - (wpred[0] - wpred[1]) + spred))
    f = smallk * (x1 - x2) + stop
    func = np.array([x3, x4, -f - x2 * bigk, f - x4 * delta])

    wpred = w
    spred = stop
    arrayofs.append(stop)

    return func


def syssolver(func, x0, t, p, dt):
    global arrayofs
    arrayofs = [spred]
    ans = [x0]
    for i in range(len(t)):
        ans.append(ans[-1] + dt * func(ans[-1], i, p))
    return ans


bigk = 10
delta = 0.5
smallk = 10
sols = [spred]

if spred == 0:
    coeffs = [1, delta, 2 * smallk, smallk * delta, smallk * bigk]
    eigs = np.roots(coeffs)
else:
    coeffs = [1, delta, 2 * (smallk + 1), (smallk + 1) * delta, (smallk + 1) * bigk]
    eigs = np.roots(coeffs)

stoptime = 5000
numpoints = stoptime*50

t = [stoptime * float(i) / float(numpoints - 1) for i in range(numpoints)]
p = [smallk, bigk, delta]
w0 = wpred
wsol = syssolver(vectorfield, w0, t, p, (t[1] - t[0]) / 10.0)

fig, ax = plt.subplots(1, 2)
plt.subplots_adjust(left=0.1, bottom=0.3)

solx = []
for i in range(0, len(wsol)):
    solx.append(wsol[i][0] - wsol[i][1])

x1_sol = []
x2_sol = []

for n in range(len(t)):
    x1_sol.append(wsol[n][0])
    x2_sol.append(wsol[n][1])

#sols = [0]
for i in range(0, len(wsol) - 1):
    sols.append(arrayofs[i])

l, = ax[0].plot(solx, sols, lw=1)
l1, = ax[0].plot(solx[0], sols[0], 'ro')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
l2, = ax[1].plot(x1_sol, x2_sol, lw=2)
#l2, = ax[1].plot(wsol[2], wsol[3], lw=2)
eq1 = -sols[-1] / smallk
l4, = ax[1].plot(eq1, 0, 'kX', lw=2)
ll = -1 / smallk
rl = -ll
l5, = ax[1].plot(ll, 0, '>m', lw=3)
l6, = ax[1].plot(rl, 0, '<m', lw=2)
l7, = ax[1].plot(x1_sol[-1], x2_sol[-1], 'rx', lw=2)
l8, = ax[1].plot(w0[0], w0[1], 'ro')
cases = ['$k+1 < K$', '$k < K < k+1$', '$K < k$', '$K = k$', '$K = k + 1$']
if bigk > smallk + 1:
    case = cases[0]
elif smallk < bigk < smallk + 1:
    case = cases[1]
elif bigk < smallk:
    case = cases[2]
elif bigk == smallk:
    case = cases[3]
elif bigk == smallk + 1:
    case = cases[4]


signs = [' + ', ' ']
if eigs[0].imag > 0:
    sign1 = signs[0]
else:
    sign1 = signs[1]

if eigs[1].imag > 0:
    sign2 = signs[0]
else:
    sign2 = signs[1]

if eigs[2].imag > 0:
    sign3 = signs[0]
else:
    sign3 = signs[1]

if eigs[3].imag > 0:
    sign4 = signs[0]
else:
    sign4 = signs[1]

if np.sqrt(wsol[0][-1] ** 2 + wsol[1][-1] ** 2) > 5:
    limit_x1 = '$\infty$'
    limit_x2 = '$\infty$'
else:
    limit_x1 = str(round(wsol[0][-1], 3))
    limit_x2 = str(round(wsol[1][-1], 3))

txt = ax[1].annotate('$\delta$ = ' + str(round(delta, 2)) + '\n'
                                                            'Init. condits.: (' + str(round(w0[0], 3)) + ', ' + str(
    round(w0[1], 3)) + ', ' + str(round(w0[2], 3)) + ', ' + str(round(w0[3], 3)) + ', $s_0:$ ' + str(
    round(arrayofs[0], 4)) + ')\n'
                       ' Eq. point: (' + str(
    round(-sols[-1] / float(smallk),
          3)) + ', 0, 0, 0)\n' + 'Limit point: (' + limit_x1 + ', ' + limit_x2 + ', 0, 0)\n' +
                     'k = ' + str(smallk) + ', K = ' + str(bigk) + '\n' +
                     case + '\n' +
                     '$\lambda_1 = $' + str(
    round(eigs[0].real, 2)) + sign1 + str(round(eigs[0].imag, 2)) + ' $\cdot$ $i$' + '\n'
                                                                                     '$\lambda_2 = $' + str(
    round(eigs[1].real, 2)) + sign2 + str(round(eigs[1].imag, 2)) + ' $\cdot$ $i$' + '\n'
                                                                                     '$\lambda_3 = $' + str(
    round(eigs[2].real, 2)) + sign3 + str(round(eigs[2].imag, 2)) + ' $\cdot$ $i$' + '\n'
                                                                                     '$\lambda_4 = $' + str(
    round(eigs[3].real, 2)) + sign4 + str(round(eigs[3].imag, 2)) + ' $\cdot$ $i$', xy=(1, 1), xytext=(-15, -15),
                     fontsize=10,
                     xycoords='axes fraction', textcoords='offset points',
                     bbox=dict(facecolor='white', alpha=0.8),
                     horizontalalignment='right', verticalalignment='top')
text = ax[0].text(solx[0], sols[0], '$s_0 = $' + str(round(arrayofs[0], 4)))

ax[0].margins(x=0)
ax[1].margins(x=0)
ax[0].set_xlabel('$x_1-x_2$', fontsize=15)
ax[0].set_ylabel('$H[x_1-x_2, s_0]$', rotation=0, fontsize=15)
ax[1].set_xlabel('$x_1$', fontsize=15)
ax[1].set_ylabel('$x_2$', rotation=0, fontsize=15)
ax[1].set_title('Projection on the phase plane ($x_1, x_2$)')
ax[0].set_xlim([-5, 5])
ax[0].set_ylim([-1.1, 1.1])
ax[0].set_title('Stop operator')
ax[0].yaxis.set_major_formatter(formatter)
ax[0].yaxis.set_label_coords(-0.15, 0.48)
ax[1].yaxis.set_label_coords(-0.1, 0.48)
x1min = -2.5
x1max = 2.5
x2min = -2
x2max = 2
ax[1].set_xlim([x1min, x1max])
ax[1].set_ylim([x2min, x2max])
ax[0].tick_params(axis="x", labelsize=15)
ax[0].tick_params(axis="y", labelsize=15)
ax[1].tick_params(axis="x", labelsize=15)
ax[1].tick_params(axis="y", labelsize=15)
ax[0].grid()
ax[1].grid()

axcolor = 'lightgoldenrodyellow'

smallkval = plt.axes([0.2, 0.2, 0.65, 0.01], facecolor=axcolor)
ssmallkval = Slider(smallkval, '$k$', 0.001, 20.000, valinit=smallk, valstep=0.001)

bigkval = plt.axes([0.2, 0.18, 0.65, 0.01], facecolor=axcolor)
sbigkval = Slider(bigkval, '$K$', 0.000, 20.000, valinit=bigk, valstep=0.001)

sval = plt.axes([0.2, 0.16, 0.65, 0.01], facecolor=axcolor)
ssval = Slider(sval, '$s_0$', -1, 1, valinit=arrayofs[0], valstep=0.01)

x1val = plt.axes([0.2, 0.14, 0.65, 0.01], facecolor=axcolor)
sx1val = Slider(x1val, '$x_{10}$', -10, 10, valinit=w0[0], valstep=0.01)

x2val = plt.axes([0.2, 0.12, 0.65, 0.01], facecolor=axcolor)
sx2val = Slider(x2val, '$x_{20}$', -10, 10, valinit=w0[1], valstep=0.01)

x3val = plt.axes([0.2, 0.1, 0.65, 0.01], facecolor=axcolor)
sx3val = Slider(x3val, '$x_{30}$', -10, 10, valinit=w0[2], valstep=0.01)

x4val = plt.axes([0.2, 0.08, 0.65, 0.01], facecolor=axcolor)
sx4val = Slider(x4val, '$x_{40}$', -10, 10, valinit=w0[3], valstep=0.01)

delval = plt.axes([0.2, 0.06, 0.65, 0.01], facecolor=axcolor)
sdelval = Slider(delval, '$\delta$', 0, 1, valinit=delta, valstep=0.01)

#timeval = plt.axes([0.2, 0.04, 0.65, 0.01], facecolor=axcolor)
#stimeval = Slider(timeval, 'Time', 0, 10000, valinit=stoptime, valstep=0.01)

#numval = plt.axes([0.2, 0.02, 0.65, 0.01], facecolor=axcolor)
#snumval = Slider(numval, 'Number of points', 0, 300000, valinit=numpoints, valstep=1)

resetax = plt.axes([0.9, 0.01, 0.025, 0.02])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def update(val):

    global arrayofs
    global wpred
    global spred
    global txt
    global text
    spred = ssval.val
    arrayofs = [spred]

    w0 = [sx1val.val, sx2val.val, sx3val.val, sx4val.val]
    wpred = w0

    if ssval.val == 0:
        coeffs = [1, sdelval.val, 2 * ssmallkval.val, ssmallkval.val * sdelval.val, ssmallkval.val * sbigkval.val]
        eigs = np.roots(coeffs)
    else:
        coeffs = [1, sdelval.val, 2 * (ssmallkval.val + 1), (ssmallkval.val + 1) * sdelval.val,
                  (ssmallkval.val + 1) * sbigkval.val]
        eigs = np.roots(coeffs)

    txt.remove()
    text.remove()
    cases = ['$k+1 < K$', '$k < K < k+1$', '$K < k$', '$K = k$', '$K = k + 1$']
    if sbigkval.val > ssmallkval.val + 1:
        case = cases[0]
    elif ssmallkval.val < sbigkval.val < ssmallkval.val + 1:
        case = cases[1]
    elif sbigkval.val < ssmallkval.val:
        case = cases[2]
    elif sbigkval.val == ssmallkval.val:
        case = cases[3]
    elif sbigkval.val == ssmallkval.val + 1:
        case = cases[4]

    signs = [' + ', ' ']
    if eigs[0].imag > 0:
        sign1 = signs[0]
    else:
        sign1 = signs[1]

    if eigs[1].imag > 0:
        sign2 = signs[0]
    else:
        sign2 = signs[1]

    if eigs[2].imag > 0:
        sign3 = signs[0]
    else:
        sign3 = signs[1]

    if eigs[3].imag > 0:
        sign4 = signs[0]
    else:
        sign4 = signs[1]

    spred = ssval.val
    arrayofs = [ssval.val]
    #stoptime = int(stimeval.val)
    #numpoints = snumval.val
    numpoints = stoptime * 50

    t = [stoptime * float(i) / float(numpoints - 1) for i in range(numpoints)]
    p = [ssmallkval.val, sbigkval.val, sdelval.val]
    wsol = syssolver(vectorfield, w0, t, p, (t[1] - t[0]) / 10.0)
    x1_sol = []
    x2_sol = []
    #x3_sol = []
    #x4_sol = []

    for n in range(len(t)):
        x1_sol.append(wsol[n][0])
        x2_sol.append(wsol[n][1])
        #x3_sol.append(wsol[n][2])
        #x4_sol.append(wsol[n][3])

    solx = []
    for i in range(0, len(wsol)):
        solx.append(wsol[i][0] - wsol[i][1])

    sols = [ssval.val]
    for i in range(0, len(wsol) - 1):
        sols.append(arrayofs[i])

    if np.sqrt(x1_sol[-1] ** 2 + x2_sol[-1] ** 2) > 5:
        limit_x1 = '$\infty$'
        limit_x2 = '$\infty$'
    else:
        limit_x1 = str(round(x1_sol[-1], 3))
        limit_x2 = str(round(x2_sol[-1], 3))
    #basin_x1 = []
    #basin_x2 = []
    #if np.sqrt((w0[0] - wsol[0][-1]) ** 2 + (w0[1] - wsol[1][-1]) ** 2) > 0.1:
    #    print('attached')
    #    basin_x1.append(w0[0])
    #    basin_x2.append(w0[1])
    #with open('basin.txt', 'a') as f:
        #for item in basin_x1:
        #    f.write("%s," % round(item, 3))
        #for item in basin_x2:
        #    f.write("%s,\r" % round(item, 3))
    txt = ax[1].annotate('$\delta$ = ' + str(round(sdelval.val, 2)) + '\n'
                                                                      'Init. condits.: (' + str(
        round(w0[0], 3)) + ', ' + str(
        round(w0[1], 3)) + ', ' + str(round(w0[2], 3)) + ', ' + str(round(w0[3], 3)) + ', $s_0:$ ' + str(
        round(ssval.val, 4)) + ')\n'
                               ' Eq. point: (' + str(
        round(-sols[-1] / float(ssmallkval.val),
              3)) + ', 0, 0, 0)\n' + 'Limit point: (' + limit_x1 + ', ' + limit_x2 + ', 0, 0)\n' +
                         'k = ' + str(round(ssmallkval.val, 2)) + ', K = ' + str(round(sbigkval.val, 2)) + '\n' +
                         case + '\n' +
                         '$\lambda_1 = $' + str(
        round(eigs[0].real, 2)) + sign1 + str(round(eigs[0].imag, 2)) + ' $\cdot$ $i$' + '\n'
                                                                                         '$\lambda_2 = $' + str(
        round(eigs[1].real, 2)) + sign2 + str(round(eigs[1].imag, 2)) + ' $\cdot$ $i$' + '\n'
                                                                                         '$\lambda_3 = $' + str(
        round(eigs[2].real, 2)) + sign3 + str(round(eigs[2].imag, 2)) + ' $\cdot$ $i$' + '\n'
                                                                                         '$\lambda_4 = $' + str(
        round(eigs[3].real, 2)) + sign4 + str(round(eigs[3].imag, 2)) + ' $\cdot$ $i$', xy=(1, 1), xytext=(-15, -15),
                         fontsize=10,
                         xycoords='axes fraction', textcoords='offset points',
                         bbox=dict(facecolor='white', alpha=0.8),
                         horizontalalignment='right', verticalalignment='top')

    text = ax[0].text(solx[0], sols[0], '$s_0 = $' + str(round(ssval.val, 4)))
    eq1 = -sols[-1] / float(ssmallkval.val)
    ll = -1 / float(ssmallkval.val)
    rl = -ll

    l.set_xdata(solx)
    l.set_ydata(arrayofs)
    l1.set_xdata(solx[0])
    l1.set_ydata(sols[0])
    l2.set_xdata(x1_sol)
    l2.set_ydata(x2_sol)
    l4.set_xdata(eq1)
    l5.set_xdata(ll)
    l6.set_xdata(rl)
    l7.set_xdata(x1_sol[-1])
    l7.set_ydata(x2_sol[-2])
    l8.set_xdata(w0[0])
    l8.set_ydata(w0[1])
    fig.canvas.draw_idle()


ssmallkval.on_changed(update)
sbigkval.on_changed(update)
ssval.on_changed(update)
sdelval.on_changed(update)
sx1val.on_changed(update)
sx2val.on_changed(update)
sx3val.on_changed(update)
sx4val.on_changed(update)


#stimeval.on_changed(update)
#snumval.on_changed(update)


def reset(event):
    ssmallkval.reset()
    sbigkval.reset()
    ssval.reset()
    sdelval.reset()
    sx1val.reset()
    sx2val.reset()
    sx3val.reset()
    sx4val.reset()
    #stimeval.reset()
    # snumval.reset()


button.on_clicked(reset)

plt.show()
