import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    cost = np.loadtxt('overall_costs.txt')
    fig = plt.figure(figsize=(10, 6))

    # cost levels of cco
    plt.plot(
        np.arange(50),
        cost[0:50, 0] / 60,
        linestyle='-',
        color='steelblue',
        markeredgecolor='steelblue',
        markerfacecolor='steelblue',
        label='60 mobile devices'
    )
    # cost levels of RS
    plt.plot(
        np.arange(50),
        cost[50:100, 0] / 80,
        linestyle='-',
        color='#996633',
        markeredgecolor='#996633',
        markerfacecolor='#996633',
        label='80 mobile devices'
    )
    # cost levels of GSC1
    plt.plot(
        np.arange(50),
        cost[100:150, 0] / 100,
        linestyle='-',
        color='#ff9999',
        markeredgecolor='#ff9999',
        markerfacecolor='#ff9999',
        label='100 mobile devices'
    )

    plt.title('Average cost in 50 time slots')
    plt.xlabel('Average cost of mobile devices')
    plt.ylabel('time slot')
    plt.legend()

    battery_fig = plt.gcf()  # 'get current figure'
    battery_fig.savefig('n.eps', format='eps', dpi=1000)
    plt.show()
