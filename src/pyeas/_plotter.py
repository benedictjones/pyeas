import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def animate(optimizer, trial_pops, bounds, func, label, save=None, algo=''):
    
    fig_ani, (ax, ax2) = plt.subplots(ncols=2, figsize=(9,4)) # , constrained_layout=True

    fig_ani.suptitle('%s on %s function' % (algo, label))

    x1 = np.linspace(bounds[0][0], bounds[0][1], 400)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = func(X1, X2)

    ax.imshow(Z, extent = np.array(bounds).flatten(), origin = 'lower', cmap = 'jet', alpha = 1)
    # ax.scatter(r1[0], r2[0], marker=".", color='r')
    it_point, = ax.plot([], [], '*', color='w', alpha=1, linestyle="None", markersize=9) #
    it_converg, = ax.plot([], [], '-o', color='w', alpha=0.45, markersize=2.5) #
    trs, = ax.plot(trial_pops[0][:,0], trial_pops[0][:,1], marker=".", color='k', linestyle="None") 
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # ax.set_xlim([2, 4.5])
    # ax.set_ylim([-0.25, 1.75])


    # ax2.set_yscale('log')
    ax2.plot(optimizer.history['best_fits'])
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Function")
    it_line, = ax2.plot([0, 0],  [np.min(optimizer.history['best_fits']), np.max(optimizer.history['best_fits'])], markersize=5, color='k', alpha=0.5) #
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    r1, r2 = zip(*optimizer.history['best_solutions'])

    def ani(i):
        # ax.clear()
        # print(i, ">", int(i/2))
        
        # ax.set_title("i: %d, i/2: %d" % (i, int(i/2)))

        if (i % 2) == 0:  # even 
            
            # print(">> trial")
            trials = trial_pops[int(i/2)]
            # trs, = ax.plot(trials[:,0], trials[:,1], marker=".", color='k') 
            trs.set_data(trials[:,0], trials[:,1]) 

        else:
            
            # print(">> parent")
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
    the_animation = animation.FuncAnimation(fig_ani, ani, frames=np.arange(len(optimizer.history['best_fits'])*2), interval=20)

    if save is not None:
        fig_path = "%s.gif" % (save)
        the_animation.save(fig_path, writer='pillow', fps=FPS, dpi=50)
    
    plt.close(fig_ani)