from objloader import OBJ,MTL
import cPickle as pickle
import os
import glob
import numpy as np
from OpenGL.GL import *
import bunch
import bisect


meshes = [os.path.split(os.path.splitext(_)[0])[1]
          for _ in glob.glob('data/meshes/*.obj')]
    

class Mesh(object):
    """ Representation of a 3D object as a triangle mesh.
        Numpy arrays for vertices and faces are easily used with opengl.
    """
    def __init__(self, vertices, faces, obj=None, scale=1.0):
        self.vertices = vertices
        self.faces = faces
        self.scale = scale
        self.obj = obj


    def draw(self):
        if obj:
            glPushMatrix()
            glScale(1.0/self.scale,1.0/self.scale,1.0/self.scale)
            glCallList(self.obj.gl_list)
            glPopMatrix()
        else:
            try:
                assert self._didwarn
            except:
                print 'TODO: implementing drawing using client arrays or buffer'


    def calculate_area_lookup(self):
        """Compute a cumsum of the surface area of the object, to help
        with random sampling
        """
        def compute_area(v1, v2, v3):
            """Triangle area formula"""
            assert v1.shape == v2.shape == v3.shape == (3,)
            c = np.cross(v2-v1, v3-v1)
            area = np.sqrt(np.dot(c,c))
            return area
        
        total_area = 0.0
        area_cumsum = []
        for tri in self.faces:
            # Compute the area of the triangle
            v1, v2, v3 = self.vertices[tri[:3],:3]
            area = compute_area(v1, v2, v3)
            total_area += area
            area_cumsum.append(total_area)

        self.area = bunch.Bunch({
            'cumsum': area_cumsum,
            'total': total_area
            })
        

    def sample_point_cloud(self, num):
        """Generate a sampling of the points from all the vertices using a
        reverse area integral
        """
        sample = np.empty((num,3), dtype='f')
        if not 'area' in self.__dict__:
            self.calculate_area_lookup()
        rnds = np.random.rand(num) * self.area.total
        inds = [bisect.bisect_left(self.area.cumsum, rnd) for rnd in rnds]
        coords = np.random.rand(num,2).astype('f')
        for i, ind, (r1,r2) in zip(range(num), inds, coords):
            v1,v2,v3 = self.vertices[self.faces[ind,:3],:3]
            if r1+r2 >= 1:
                r1 = 1-r1
            point = v1 + (v2-v1)*r1 + (v3-v1)*r2
            sample[i,:] = point
        return sample


def load_random():
    import random
     return load(random.choice(meshes))


def load(meshname, do_scale=False):
    if meshname in ('teapot', 'bunny69k', 'blueghost'):
        do_scale = True

    # Load a mesh from an obj file
    global obj
    savedpath = os.getcwd()

    def build(src_file):
        savedpath = os.getcwd()
        try:
            os.chdir('data/meshes')
            return OBJ('%s.obj' % meshname)
        finally:
            os.chdir(savedpath)
    obj = cache_or_build('data/meshes/%s.obj' % meshname,
                         'data/meshes/%s_cache.pkl' % meshname,
                         build)
    try:
        os.chdir('data/meshes')
        obj.compile()
    finally:
        os.chdir(savedpath)

    global points
    points = np.array(obj.vertices,'f')

    # Scale the model down and center it
    global scale
    scale = 1.0
    if do_scale:
        scale = points.std()*2
        points /= scale

    points = np.hstack((points,np.zeros((points.shape[0],1),'f')))

    # Just the vertices, useful for putting in a vertexbuffer
    global vertices
    vertices = np.ascontiguousarray(points)

    # The faces, specifically triangles, as vertex indices
    global faces
    faces = np.array([(v[0],v[1],v[2],0) for v,_,_,_ in obj.faces])-1

    obj = Mesh(vertices, faces, obj, scale)
    return obj


def cache_or_build(src_file, cache_file, build_func):
    try:
        t1 = os.stat(src_file).st_mtime
        t2 = os.stat(cache_file).st_mtime
        assert t1 < t2
        print 'Loading from cache %s' % cache_file
        with open(cache_file, 'rb') as f:
            obj = pickle.load(f)
    except (OSError, IOError, AssertionError):
        print 'Building %s -> %s' % (src_file, cache_file)
        obj = build_func(src_file)
        print 'Saving to cache %s' % cache_file
        with open(cache_file, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj
