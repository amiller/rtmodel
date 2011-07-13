import numpy as np
import transformations


def random_pose(radius):
    """Returns:
        Quaternion shape==(4,), Transformation shape==(3,)
    """
    return transformations.random_quaternion(), (np.random.rand(3)*2-1) * radius


def random_sequence(duration=10.0, poses=5, radius=1.0, fps=30.0):
    """
    Args:
        duration: in seconds
        poses: number of different target poses
        radius: random translation within a box size [-radius, radius]
        fps: 1./seconds
    """
    # First pose is identity quat, identity translation
    Start = np.array([1,0,0,0], 'f'), np.array([0,0,0])
    ts = 0.0
    matrices = []
    timestamps = []
    dt = 1. / fps    
    for i in range(poses):
        # New target pose
        End = random_pose(radius)
        frac = 0.0
        while not np.allclose(frac, 1.0):
            RT = np.eye(4, dtype='f')

            # Rotation
            Q = transformations.quaternion_slerp(Start[0], End[0], frac)
            RT[:3,:3] = transformations.quaternion_matrix(Q)[:3,:3]

            # Translation
            T = (1-frac)*Start[1] + (frac)*End[1]
            RT[:3,3] = T

            timestamps.append(ts)
            matrices.append(RT)

            ts += dt
            frac += dt / (duration / poses)

        Start = End

    return PoseSequence(np.array(matrices), np.array(timestamps))


class PoseSequence(object):
    def __init__(self, matrices, timestamps):
        assert matrices.dtype == np.float32
        assert matrices.shape[1:3] == (4, 4)
        assert timestamps.shape[0] == matrices.shape[0]
        self.matrices = matrices
        self.timestamps = timestamps

    def __iter__(self):
        return iter(zip(self.timestamps, self.matrices))
