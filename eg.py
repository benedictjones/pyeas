import numpy as np
from pyeas import DE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import matplotlib
# matplotlib.use("Agg")


def f(x1, x2):
    return 0.5*x1**2 + (5/2)*x2**2 - x1*x2 - 2*(x1 + x2)


optimizer = DE(mut=0.6,
               crossp=0.6,
               bounds=np.array([[-10,10],[-10,10]]),
               #groupings=[2,4],
               population_size=20,
               mut_scheme = 'best1',  # 'ttb1', rand1
               seed=1)

trial_pops = []
num_gens = 40
for generation in range(num_gens):
    # print("Gen:", generation)
    solutions = []
    
    # Ask a parameter
    trial_pop = optimizer.ask(loop=generation)
    trial_pops.append(trial_pop)
    # print(trial_pop)

    for j, trial in enumerate(trial_pop):
        value = f(trial[0], trial[1])
        solutions.append((value))
        #print(f"#{generation} {value} (x1={trial[0]}, x2 = {trial[1]}))")

    # Tell evaluation values.
    optimizer.tell(solutions, trial_pop)

    print("Gen:", generation, optimizer.best_member[0])

fig = plt.figure()
plt.plot(optimizer.history['best_fits'])
# plt.yscale("log")


print(optimizer.history['best_solutions'][-1])


x1 = np.linspace(1.5, 4, 150)
x2 = np.linspace(0, 1.5, 150)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)


fig = plt.figure(figsize = (10,7))
contours = plt.contour(X1, X2, Z, 20)
plt.clabel(contours, inline = True, fontsize = 10)
plt.title("Evolution of the cost function during gradient descent", fontsize=15)

r1, r2 = zip(*optimizer.history['best_solutions'])
plt.plot(r1, r2, label='OAIES Solution')
plt.plot(r1, r2, '*')
plt.legend()

# plt.show()
# exit()







fig_ani, (ax, ax2) = plt.subplots(ncols=2, figsize=(9,4))

fig_ani.suptitle('OpenAI-ES on $f = 0.5*x1^2 + (5/2)*x2^2 - x1*x2 - 2*(x1 + x2)$')

x1 = np.linspace(-5, 5, 400)
x2 = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

ax.imshow(Z, extent = [-5,5,-5,5], origin = 'lower', cmap = 'jet', alpha = 1)
# ax.scatter(r1[0], r2[0], marker=".", color='r')
it_point, = ax.plot([], [], '*', color='w', alpha=1, linestyle="None") #
it_converg, = ax.plot([], [], '-*', markersize=5, color='w', alpha=0.3) #
trs, = ax.plot(trial_pops[0][:,0], trial_pops[0][:,1], marker=".", color='k', linestyle="None") 
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xlim([1.75, 5])
ax.set_ylim([-0.25, 2])


# ax2.set_yscale('log')
ax2.plot(optimizer.history['best_fits'])
ax2.set_xlabel("Generation")
ax2.set_ylabel("Function")
it_line, = ax2.plot([0, 0],  [np.min(optimizer.history['best_fits']), np.max(optimizer.history['best_fits'])], markersize=5, color='k', alpha=0.5) #
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def ani(i):
    # ax.clear()
    #print(i, ">", int(i/2))
    
    # ax.set_title("i: %d, i/2: %d" % (i, int(i/2)))

    if (i % 2) == 0:  # even 
        
        #print(">> trial")
        trials = trial_pops[int(i/2)]
        # trs, = ax.plot(trials[:,0], trials[:,1], marker=".", color='k') 
        trs.set_data(trials[:,0], trials[:,1]) 

    else:
        
        #print(">> parent")
        trs.set_data([], []) 
        # if int(i/2) > 0:
        #     trs.remove()

        # it_point.set_xdata(r1[int(i/2)])
        # it_point.set_ydata(r2[int(i/2)])
        it_point.set_data(r1[int(i/2)], r2[int(i/2)])
        it_converg.set_data(r1[:int(i/2)], r2[:int(i/2)])
        # ax.plot
        it_line.set_xdata([int(i/2), int(i/2)])
    



FPS = 20 # num_gens/300
the_animation = animation.FuncAnimation(fig_ani, ani, frames=np.arange(num_gens*2), interval=20)

print("DONE")
plt.show()
