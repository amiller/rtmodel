from wxpy3d import PointWindow
from OpenGL.GL import *
from rtmodel import mesh
from rtmodel import offscreen
from rtmodel import camera
from rtmodel import rangeimage
import rtmodel.rangeimage_speed
from rtmodel import rtmodel
from rtmodel import pointmodel
from rtmodel import transformations
from rtmodel import fasticp
from rtmodel import posesequence
import pylab


if not 'window' in globals():
    window = PointWindow(size=(640,480))
    window.Move((632,100))
    print """
    Demo Objrender:
        search for "Points to draw" and uncomment different points
        refresh()
        perturb(): apply a random matrix to the point model
        load_obj(): select a random object and load it
    """


def random_perturbation(angle_max=np.pi/100, trans_max=0.05, point=None):
    axis = np.random.rand(3)-0.5
    angle = np.random.rand()*angle_max
    trans = (np.random.rand(3)*2-1)*trans_max
    m4 = transformations.rotation_matrix(angle, axis)
    m4[:3,3] += trans
    return m4


def load_obj(name='blueghost'):
    global obj, points, points_p, range_image, points_range, pnew

    #obj = mesh.load_random()
    window.canvas.SetCurrent()
    obj = mesh.load(name)

    obj.calculate_area_lookup()
    obj.RT = np.eye(4, dtype='f')
    obj.RT[:3,3] = -obj.vertices[:,:3].mean(0)
    obj.RT[:3,3] += [0,0,-3.0]

    points = obj.sample_point_cloud(10000)

    # Range image of the original points
    range_image = obj.range_render(camera.kinect_camera())
    range_image.compute_normals()
    points_range = range_image.point_model()
    
    pnew = points_p = points
    #range_image = obj.range_render(np.dot(camera.kinect_camera(), rp))

    window.lookat = obj.RT[:3,3] + obj.vertices[:,:3].mean(0)
    window.Refresh()
    

def perturb(max_iters=100, mod=10):
    global pnew, uv, err, points_range, rimg, range_image

    # Apply a perturb to points_p
    obj.RT = np.eye(4, dtype='f')
    obj.RT[:3,3] = -obj.vertices[:,:3].mean(0)
    obj.RT[:3,3] += [0,0,-3.0]

    # Rotated object view
    RT = obj.RT
    rp = random_perturbation().astype('f')
    obj.RT = np.dot(rp, obj.RT)
    range_image = obj.range_render(camera.kinect_camera())
    obj.RT = RT

    points_range = range_image.point_model(True)

    # Original object view
    rimg = obj.range_render(camera.kinect_camera())
    pnew = rimg.point_model()

    # Estimate the transformation rp
    for iters in range(max_iters):
        pnew, err, npairs, uv = fasticp.fast_icp(range_image, pnew, 0.1, dist=0.02)
        if iters % mod == 0 or 1:
            #print '%d iterations, [%d] RMS: %.3f' % (iters, npairs, np.sqrt(err))
            window.Refresh()
            pylab.waitforbuttonpress(0.02)

    window.Refresh()


def start():
    global seqiter, model
    seqiter = iter(posesequence.random_sequence())
    model = rtmodel.RTModel()


def go():
    start()
    while 1:
        once()
        window.Refresh()
        pylab.waitforbuttonpress(0.02)


def once():
    global range_image, points_range
    ts, M = seqiter.next()

    # Take the image from an alternate camera location
    obj.RT = np.dot(M, obj.RT)
    range_image = obj.range_render(camera.kinect_camera())
    points_range = range_image.point_model()
    range_image.camera.RT = np.eye(4, dtype='f')
        
    #model.add(rimg)


def animate_random(max_iters=1000, mod=100):
    global pnew, points_range
    # Apply a perturb to points_p
    obj.RT = np.eye(4, dtype='f')
    obj.RT[:3,3] = -obj.vertices[:,:3].mean(0)
    obj.RT[:3,3] += [0,0,-3.0]
    RT = obj.RT

    prev_rimg = obj.range_render(camera.kinect_camera())
    window.canvas.SetCurrent()
    pnew = prev_rimg.point_model(True)
    points_range = pnew

    if 0:
        obj.RT = np.dot(RT, M)
        rimg = obj.range_render(camera.kinect_camera())
        window.canvas.SetCurrent()
        pm = rimg.point_model(True)
        points_range = pm

        for iters in range(max_iters):
            pnew, err, npairs, uv = fasticp.fast_icp(rimg, pnew, 1000, dist=0.005)
            if iters % mod == 0:
                # print '%d iterations, [%d] RMS: %.3f' % (iters, npairs, np.sqrt(err))
                window.Refresh()
                pylab.waitforbuttonpress(0.02)

        pnew = pm

        window.Refresh()
        pylab.waitforbuttonpress(0.02)        


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
    # Perturbed points (previous estimate)
    glColor(0,1,0)
    glPointSize(1)
    points_p.draw()

    # Red: points from the range image (observation)
    glColor(1,0,0)
    glPointSize(1)    
    points_range.draw()

    # Estimated points (corrected estimate)
    glColor(1,1,0)
    glPointSize(2)
    pnew.draw()


    global projcam, modelcam
    modelcam = glGetFloatv(GL_MODELVIEW_MATRIX).transpose()
    projcam = glGetFloatv(GL_PROJECTION_MATRIX).transpose()
    #obj.draw()
    glDisable(GL_LIGHTING)
    glColor(1,1,1,1)


window.Refresh()
