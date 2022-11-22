from easydict import EasyDict as edict

net_param = edict({'rows': 5,
                   'cols': 5,
                   'frac_of_static_elements': .3,
                   'weight_init': None,  #'rand',
                   'seed': 2})

mem_param = edict({'kp0': 2.555173332603108574e-06,  # model kp_0
                   'kd0': 6.488388862524891465e+01,  # model kd_0
                   'eta_p': 3.492155165334443012e+01,  # model eta_p
                   'eta_d': 5.590601016803570467e+00,  # model eta_d
                   'g_min': 1.014708121672117710e-03,  # model g_min
                   'g_max': 2.723493729125820492e-03,  # model g_max
                   'g0': 1.014708121672117710e-03  # model g_0
                   })

sim_param = edict({'T': 100e-3, #4e-3, # [s]
                    #'steps': 100,
                   'sampling_rate': 500 #1000 # [Hz]  # =steps / T  # [Hz]
                    #dt = T / steps  # [s] or    dt = 1/sampling_rate bc  steps= sampling_rate *T
                    })

volt_param = edict({'VMAX': 20,
                     'VMIN': 1,
                     'VSTART': .3, #'rand',
                     'amplitude_gain_factor': .01
                     })

train_param = edict({'Horizon': 300,
                     'MAX_TRAJECTORIES': 1000,
                     'learning_rate': 3e-3,
                     'gamma': .99,
                     #'eps': 1e-4,
                     #'desired_state': 2.9e-3  # [S] # Corrisponde a un voltaggio 1.5V})
                     })

env_param = edict({'eps': 1e-3,
                   'desired_state': 2.5e-3  # [S] # Corrisponde a un voltaggio 1.5V})
                   })

# t1_list = np.arange(0, T1, dt) # [s]
# t2_list = np.arange(T1, T1+T2, dt)[1:] # [s]
# t3_list = np.arange(T1+T2, T, dt)[1:] # [s]

# t_list = np.hstack((t1_list, t2_list, t3_list))
#t_list = np.arange(0, T + 1 / sampling_rate, 1 / sampling_rate)  # [s]
