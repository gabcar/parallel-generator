"""augmentation.py
"""
import scipy.ndimage
import numpy as np

def rotation(X, Y, param, **kwargs):
    """
    Rotates image param degrees
    """
    angle = np.random.uniform(-param, param)

    offset = X.min()
    x = X - offset
    # order 3 = cubic spline
    x = scipy.ndimage.rotate(x, angle, order=3, reshape=False)

    x = x + offset
    # order 0 = nearest neighbour interpolation
    y = [scipy.ndimage.rotate(label, angle, order=0, reshape=False)
         for label in Y]

    return x, y

def scaling(X, Y, param, **kwargs):
    pass

def affine_transform(X, Y, param, **kwargs):
    if isinstance(param['scaling'], float) or isinstance(param['scaling'], int):
        scaling = [-param['scaling'], param['scaling']]
    else:
        scaling = param['scaling']

    deg = np.random.uniform(-param['rotation'], param['rotation'])
    rad = deg * np.pi * 2 / 360
    scale =  1 + np.random.uniform(*scaling)
    translation =  np.random.uniform(-param['translation'], param['translation'])
    
    x = X - X.min()
    shape = np.array(X.shape)
    M = build_affine_matrix(scale, - shape / 2, rad)
    M = np.linalg.inv(M)

    x = scipy.ndimage.affine_transform(x.squeeze(), M)

    y = []
    for _y in Y:
        order = dtype_to_order(_y.dtype)
        y.append(scipy.ndimage.affine_transform(_y.squeeze(), M, order=order, mode='nearest'))

    x = x + X.min()

    return x, y

def random_brightness(X, Y, param, **kwargs):
     intensity = np.random.uniform(-param, param)

     x = X + intensity

     return x, Y

def random_contrast(X, Y, param, **kwargs):
     gamma = np.random.uniform(param['lower'], param['upper'])

     delta = X.min()

     x = X - delta
     # print(max_val)

     x = x.max() * ((x / x.max()) ** gamma)
     x += delta
     return x, Y

def random_flip(X, Y, param, **kwargs):
     if np.random.random() < param:
          axis = np.random.randint(0, 2)

          x = np.flip(X, axis=axis)
          y = [np.flip(_y, axis=axis) for _y in Y]
     else:
          x = X
          y = Y
          
     return x, y

def flip(X, Y, param, **kwargs):
    if np.random.random() < param['frequency']:
        x = np.flip(X, axis=param['axis'])
        y = [np.flip(_y, axis=param['axis']) for _y in Y]
    else:
        x = X
        y = Y
    assert np.sum(X) == np.sum(x)   
    return x, y


def build_affine_matrix(s, c, rad, t=(0, 0)):
    if isinstance(s, float):
        s = [s, s]

    C = [[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1]]

    S = [[s[0], 0, 0], [0, s[1], 0], [0, 0, 1]]

    T = [[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]
    
    R = [[np.cos(rad), -np.sin(rad), 0],
         [np.sin(rad), np.cos(rad), 0],
         [0 , 0, 1]]
    
    C_prime = [[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]]

    M = np.dot(C_prime, np.dot(T, np.dot(R, np.dot(S, C))))

    return M

def centered_affine_transform(X, Y, param, **kwargs):

    deg = np.random.uniform(-param['rotation'], param['rotation'])
    rad = deg * np.pi * 2 / 360
    scale =  1 + np.random.uniform(-param['scaling'], param['scaling'])
    translation =  np.random.uniform(-param['translation'], param['translation'])
    
    x = X - X.min()
    
    shape = np.array(X.shape)

    if Y[0].any():
        center = - np.mean(np.where(Y[0]), axis=1)
    else: 
        center = shape * (-1) / 2

    M = build_affine_matrix(scale, center, rad)

    x = scipy.ndimage.affine_transform(x.squeeze(), M)
    y = [scipy.ndimage.affine_transform(_y.squeeze(), M, order=0) for _y in Y]

    x = x + X.min()

    return x, y


def dtype_to_order(dtype):
    if dtype in [np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 bool, np.bool]:
        return 0
    return 3

def affine_transform_landmark(X, Y, param, **kwargs):
    lm_size = Y[0].shape
    deg = np.random.uniform(-param['rotation'], param['rotation'])
    rad = deg * np.pi * 2 / 360
    scale = 1 + np.random.uniform(-param['scaling'], param['scaling'], size=lm_size)
    
    translation = np.random.uniform(-param['translation'], param['translation'], size=lm_size)
    
    x = X - X.min()
    shape = np.array(X.shape)

    M = build_affine_matrix(scale, - shape / 2, rad, translation)
    
    x = scipy.ndimage.affine_transform(x.squeeze(), np.linalg.inv(M))

    y = np.pad(Y[0], (0, 1), mode='constant')
    y[-1] = 1

    y = np.dot(M, y)
    y = [y[:2]]

    return x, y
