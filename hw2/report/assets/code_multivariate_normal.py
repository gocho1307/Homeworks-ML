from scipy.stats import multivariate_normal

x8 = multivariate_normal.pdf([0.38, 0.52], mean=[0.4414, 0.41],
                             cov=[[0.0491, -0.0211], [-0.0211, 0.0375]])
x9 = multivariate_normal.pdf([0.42, 0.59], mean=[0.4414, 0.41],
                             cov=[[0.0491, -0.0211], [-0.0211, 0.0375]])

x8_A = multivariate_normal.pdf([0.38, 0.52], mean=[0.24, 0.52],
                               cov=[[0.0064, 0.0096], [0.0096, 0.0336]])
x8_B = multivariate_normal.pdf([0.38, 0.52], mean=[0.5925, 0.3275],
                               cov=[[0.0229, -0.0098], [-0.0098, 0.03149]])
x9_A = multivariate_normal.pdf([0.42, 0.59], mean=[0.24, 0.52],
                               cov=[[0.0064, 0.0096], [0.0096, 0.0336]])
x9_B = multivariate_normal.pdf([0.42, 0.59], mean=[0.5925, 0.3275],
                               cov=[[0.0229, -0.0098], [-0.0098, 0.03149]])
