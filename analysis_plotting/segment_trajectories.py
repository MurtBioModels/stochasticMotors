import numpy as np


def diff_asc(x_list, t_list):

    runs_per_it = []
    t_per_it = []

    for index, it_list in enumerate(x_list):
        #print('forloop x_list')
        #print(f'len(runs_per_it)={len(runs_per_it)}')
        #print(f'len(t_per_it)={len(t_per_it)}')
        diff_x = np.diff(it_list)
        diff_t = np.diff(t_list[index])
        #print(diff_x)
        #print(diff_t)
        runs = []
        timepoints = []
        asc_count = 0
        t_count = 0

        for index, x in enumerate(diff_x):
            #print('forloop diff_x')
            #print(f'x={x}')
            #print(f'diff_t[index]={diff_t[index]}')
            #print(f'asc_count={asc_count}')
            #print(f't_count={t_count}')
            #print(f'len(run)={len(runs)}')
            #print(f'len(timepoints)={len(timepoints)}')
            if x >= 0 and x <= 8.005:
                #print('if x >= 0 and x <= 8.005')
                asc_count += x
                t_count += diff_t[index]
            else:
                #print(f'< 0 happend')
                if asc_count > 0:
                    #print('if asc_count > 0')
                    runs.append(asc_count)
                    timepoints.append(t_count)
                    asc_count = 0
                    t_count = 0
                else:
                    #print('else')
                    asc_count = 0
                    t_count = 0

        if asc_count > 0:
            #print('if asc_count > 0 na diff_x')
            runs.append(asc_count)
            timepoints.append(t_count)
            #print(f'len(run)={len(runs)}')
            #print(f'len(timepoints)={len(timepoints)}')

        runs_per_it.append(runs)
        t_per_it.append(timepoints)

    return runs_per_it, t_per_it

def diff_unbind(x_list):

    back_movement = []

    for index, it_list in enumerate(x_list):
        diff_x = np.diff(it_list)
        #print(diff_x)
        for index, x in enumerate(diff_x):
            #print(f'x={x}')
            #print(f'diff_t[index]={diff_t[index]}')
            #print(f'asc_count={asc_count}')
            #print(f't_count={t_count}')
            if x > 8 or x < -8:
                back_movement.append(x)
            else:
                pass

    return back_movement
def diff_desc(x_list, t_list):

    runs_per_it = []
    t_per_it = []

    for index, it_list in enumerate(x_list):
        diff_x = np.diff(it_list)
        diff_t = np.diff(t_list[index])
        #print(diff_x)
        runs = []
        timepoints = []
        desc_count = 0
        t_count = 0
        for index, x in enumerate(diff_x):
            #print(f'x={x}')
            #print(f'diff_t[index]={diff_t[index]}')
            #print(f'desc_count={desc_count}')
            #print(f't_count={t_count}')
            if x <= 0 and x >= -8:
                desc_count += x
                t_count += diff_t[index]
            else:
                #print(f'> 0 happend')
                if desc_count < 0:
                    runs.append(desc_count)
                    timepoints.append(t_count)
                    desc_count = 0
                    t_count = 0
                else:
                    desc_count = 0
                    t_count = 0

        if desc_count < 0:
            runs.append(desc_count)
            timepoints.append(t_count)


        runs_per_it.append(runs)
        t_per_it.append(timepoints)

    return runs_per_it, t_per_it


def segment_parratio_test(xb, t):
    """

    Parameters
    ----------
    Check

    Returns
    -------

    """


    runs_asc, t_asc = diff_asc(x_list=xb, t_list=t)
    print(f'runs_asc={runs_asc}')
    print(f't_asc={t_asc}')
    flat_runs_asc = [element for sublist in runs_asc for element in sublist]
    print(f'flat_runs_asc={flat_runs_asc}')
    flat_t_asc = [element for sublist in t_asc for element in sublist]
    print(f'flat_t_asc={flat_t_asc}')
    v_asc =  [i/j for i, j in zip(flat_runs_asc, flat_t_asc)]
    print(f'v_asc={v_asc}')

    runs_desc, t_desc = diff_desc(x_list=xb, t_list=t)
    print(f'runs_desc={runs_desc}')
    print(f't_desc={t_desc}')
    flat_runs_desc = [element for sublist in runs_desc for element in sublist]
    print(f'flat_runs_desc={flat_runs_desc}')
    flat_t_desc = [element for sublist in t_desc for element in sublist]
    print(f'flat_t_desc={flat_t_desc}')
    v_desc = [i/j for i, j in zip(flat_runs_desc, flat_t_desc)]
    print(f'v_desc={v_desc}')

    return

if __name__ == "__main__":
    x_list = [[0,1,2, 2, 3         ,11,    20,  22, 2,3,2,2,1,0,-1,-2,-1,0,1], [0,1,2,5,3,2,9,2,2,1,0,-1,-2,-1,0,1]]
    t_list = [[0, 0.1, 0.2, 0.3, 0.7, 1,    1.2, 1.4, 1.5, 1.6, 1.9, 2.1, 3, 3.1, 3.3, 3.5, 4.1,4.2, 4.4], [0, 0.1, 0.2, 0.9, 1, 1.2, 1.5, 1.6, 1.9, 2.1, 3, 3.1, 3.3, 3.5, 4.1,4.2]]
    xs, ts = diff_asc(x_list, t_list)
    xs2, ts2 = diff_desc(x_list, t_list)
    print(xs)
    print(ts)
    print(xs2)
    print(ts2)
    back = diff_unbind(x_list)
    print(back)



