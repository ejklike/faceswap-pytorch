import numpy as np

from .umeyama import umeyama

# without jawline
mean_face_x_51 = np.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689])

mean_face_y_51 = np.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182])

mean_landmark_51 = np.stack([mean_face_x_51, mean_face_y_51], axis=1)

# with jawline
mean_face_x_68 = [
    0.07923969, 0.08292195, 0.09679271, 0.12214152, 0.16868786, 0.23978939,
    0.32566245, 0.42231828, 0.5317778,  0.6412963,  0.73810587, 0.82444436,
    0.89479268, 0.93939549, 0.96111934, 0.97057984, 0.97119327, 0.16384622,
    0.21780355, 0.29129935, 0.36746024, 0.43929451, 0.58644596, 0.66015267,
    0.73746645, 0.81323655, 0.87075719, 0.51534534, 0.51622145, 0.51711886,
    0.5181643,  0.43370116, 0.47550124, 0.52071293, 0.56587411, 0.607054,
    0.25241872, 0.29866302, 0.35574972, 0.40371898, 0.35250718, 0.29679176,
    0.63132608, 0.67907338, 0.73597236, 0.78286538, 0.74031227, 0.6849985,
    0.35316776, 0.41458778, 0.47767765, 0.5227329,  0.56983206, 0.63519581,
    0.69951672, 0.63944716, 0.57641051, 0.52539841, 0.47641546, 0.41379549,
    0.38008479, 0.477956,   0.52338979, 0.57105779, 0.67240914, 0.57253962,
    0.52401065, 0.47756123]
mean_face_y_68 = [
    0.33922374, 0.45695537, 0.57564802, 0.6919216,  0.80034126, 0.8957325,
    0.97706876, 1.04329,    1.06080371, 1.03981924, 0.97226883, 0.88962408,
    0.79249416, 0.68154664, 0.56223825, 0.44175893, 0.32211874, 0.24915174,
    0.20425586, 0.19236732, 0.20358221, 0.2331356,  0.22814164, 0.19592384,
    0.18236098, 0.19282801, 0.23529338, 0.31863546, 0.39620045, 0.47379769,
    0.5531578,  0.60405446, 0.62076344, 0.63426822, 0.61879658, 0.60157672,
    0.33105226, 0.30264635, 0.30302065, 0.33867711, 0.34998762, 0.35047898,
    0.33413667, 0.29645404, 0.29472129, 0.32130528, 0.34184938, 0.34373433,
    0.74618916, 0.71905384, 0.70683589, 0.71709228, 0.70541448, 0.71565573,
    0.73941919, 0.80523688, 0.83543667, 0.84170638, 0.83750591, 0.8100456,
    0.7499796,  0.74513235, 0.7489243,  0.74332895, 0.74417703, 0.77660929,
    0.78337078, 0.77847635]

# with jawline
mean_landmark_68 = np.stack([mean_face_x_68, mean_face_y_68], axis=1)

def get_align_mat(landmark, jawline=False):
    """
    return affine transformation matrix of size (2, 3)
    """
    if jawline:
        return umeyama(landmark, mean_landmark_68, True)[0:2]
    else:
        return umeyama(landmark[17:], mean_landmark_51, True)[0:2]