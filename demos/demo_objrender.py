from wxpy3d import PointWindow
from OpenGL.GL import *
from rtmodel import mesh
from rtmodel import offscreen
from rtmodel import camera
from rtmodel import rangeimage
from rtmodel import pointmodel
from rtmodel import transformations
from rtmodel import fasticp
from rtmodel import volume
from rtmodel import opencl
from rtmodel import cuda

if not 'window' in globals():
    window = PointWindow(size=(640,480))#, pos=(20,20))
    print """
    Demo Objrender:
        refresh()
        load_obj(): select a random object and load it
    """

def load_obj(name='blueghost'):
    global obj, points, range_image, vol

    window.canvas.SetCurrent()
    obj = mesh.load(name)

    obj.RT = np.eye(4, dtype='f')
    obj.RT[:3,3] = -obj.vertices[:,:3].mean(0)

    # Range image of the original points
    cam = camera.kinect_camera()
    cam.RT[:3,3] += [0, 0, 3.0]
    range_image = obj.range_render(cam)
    points = range_image.point_model(True)
    
    #range_image = obj.range_render(np.dot(camera.kinect_camera(), rp))

    vol = volume.Volume(N=128)
    vol.distance_transform_cuda(range_image)

    window.lookat = obj.RT[:3,3] + obj.vertices[:,:3].mean(0)
    window.Refresh()


def raycast():
    vol.raycast_cuda(windowcam)
    figure(1);
    clf();
    title('Depth (volumetric raycast, cuda)');
    imshow(vol.cuda_volume.c_depth);
    figure(2);
    clf();
    title('Normals (volumetric raycast, cuda)');
    imshow(vol.cuda_volume.c_norm*.5+.5);

def refresh():
    global obj, points
    window.Refresh()

@window.event
def pre_draw():
    global obj
    if not 'obj' in globals():
        load_obj()
        window.Refresh()
    
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
    points.draw()
    glColor(1,1,0)

    glLineWidth(2)
    range_image.camera.render_frustum()

    global windowcam
    modelcam = glGetFloatv(GL_MODELVIEW_MATRIX).transpose()
    projcam = glGetFloatv(GL_PROJECTION_MATRIX).transpose()
    windowcam = camera.Camera(range_image.camera.KK, np.linalg.inv(modelcam))
    obj.draw()
    glDisable(GL_LIGHTING)
    glColor(1,1,1,1)

    vol.render_bounds()

window.Refresh()
