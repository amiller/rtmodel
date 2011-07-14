from wxpy3d import PointWindow
from OpenGL.GL import *
from rtmodel import mesh
from rtmodel import offscreen
from rtmodel import camera
from rtmodel import rangeimage
from rtmodel import pointmodel
from rtmodel import transformations
from rtmodel import fasticp


if not 'window' in globals():
    window = PointWindow(size=(640,480))
    print """
    Demo Objrender:
        search for "Points to draw" and uncomment different points
        refresh()
        perturb(): apply a random matrix to the point model
        load_obj(): select a random object and load it
    """


def random_perturbation(angle_max=np.pi/10, trans_max=0.10, point=None):
    axis = np.random.rand(3)-0.5
    angle = np.random.rand()*angle_max
    trans = (np.random.rand(3)*2-1.0)*trans_max
    m4 = transformations.rotation_matrix(angle, axis)
    #m4[:3,3] += trans
    return m4


def load_obj():
    global obj, points, points2, points3, range_image

    #obj = mesh.load_random()
    obj = mesh.load('blueghost')
    obj.calculate_area_lookup()
    obj.RT[:3,3] = -obj.vertices[:,:3].mean(0)
    obj.RT[:3,3] += [0,0,-3.0]

    points = obj.sample_point_cloud(10000)
    n = np.hstack((points.norm, np.ones((points.norm.shape[0],1),'f')))
    dummy = np.array(len(points.xyz)*[[1,1,1,1]],'f')

    range_image = obj.range_render(camera.kinect_camera())
    range_image.compute_normals()
    points2 = range_image.point_model()

    window.lookat = obj.RT[:3,3] + obj.vertices[:,:3].mean(0)
    refresh()


def perturb():
    global obj, points, points2, points3, range_image
    rp = random_perturbation()
    points3 = pointmodel.PointModel(points.xyz, points.norm, np.dot(points.RT, rp))
    window.Refresh()
    

def refresh():
    global obj, points, points2, points3, range_image    
    points3 = pointmodel.PointModel(points.xyz, points.norm, points.RT)
    window.Refresh()


if not 'obj' in globals():
    load_obj()


@window.event
def pre_draw():
    glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.3, 0.3, 0.3, 0.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    #glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_MODELVIEW)


@window.event
def post_draw():
    # Points to draw
    
    glColor(0,1,0)
    glPointSize(4)
    #points.draw()
    glColor(1,1,0)
    points3.draw()

    # Red: points from the range image
    glColor(1,0,0)
    points2.draw()

    global projcam, modelcam
    modelcam = glGetFloatv(GL_MODELVIEW_MATRIX).transpose()
    projcam = glGetFloatv(GL_PROJECTION_MATRIX).transpose()
    obj.draw()
    glDisable(GL_LIGHTING)
    glColor(1,1,1,1)


window.Refresh()

