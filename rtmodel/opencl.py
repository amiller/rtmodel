from OpenGL.GL import *
from OpenGL.GLU import *
import calibkinect
import pyopencl as cl
import numpy as np


def print_info(obj, info_cls):
    for info_name in sorted(dir(info_cls)):
        if not info_name.startswith("_") and info_name != "to_string":
            info = getattr(info_cls, info_name)
            try:
                info_value = obj.get_info(info)
            except:
                info_value = "<error>"

            print "%s: %s" % (info_name, info_value)

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context(devices=[device])
queue = cl.CommandQueue(context)
mf = cl.mem_flags
sampler = cl.Sampler(context, True,
                     cl.addressing_mode.CLAMP,
                     cl.filter_mode.LINEAR)


kernel_source_template = """

__constant float EPSILON = 1e-5;
__constant float TAU = 6.2831853071;
__constant int FILT = 3;
__constant float DIST = 0.2; // Maximum distance away from the axis
__constant float N_WIDTH = 512;

inline float4 matmul3(const float4 mat[3], const float4 r1) {
  return (float4)(dot(mat[0],r1),dot(mat[1],r1),dot(mat[2],r1), 0);
}
inline float4 matmul4h(const float4 mat[4], const float4 r1) {
  float W = 1.0 / dot(mat[3],r1);
  return (float4)(W*dot(mat[0],r1),W*dot(mat[1],r1),W*dot(mat[2],r1), 1);
}

kernel void integrate_tsdf(
        global short2 *voxels,
        global const float *depth,
        float4 m0, float4 m1, float4 m2, float4 m3
)
{
  unsigned int iX = get_global_id(0);
  unsigned int iY = get_global_id(1);
  unsigned int iZ = get_global_id(2);
  unsigned int index = iX*(N_WIDTH*N_WIDTH) + iY*(N_WIDTH) + iZ;
  
  float4 XYZ_ = (float4)(iX+0.5, iY+0.5, iZ+0.5, 1);

  float4 mat[4];
  mat[0] = m0;
  mat[1] = m1;
  mat[2] = m2;
  mat[3] = m3;

  float4 xyz_ = matmul4h(mat, XYZ_);

  // Expected depth for this voxel in meters
  float dref = 1.0 / xyz_.z;

  int ix = floor(xyz_.x);
  int iy = floor(xyz_.y);

  if (ix < 0 || ix >= 640) return;
  if (iy < 0 || iy >= 480) return;

  float d = depth[iy*640 + ix];
  float sd = d-dref;

  // Get the values before integrating
  short2 sw = voxels[index];
  ushort w = as_ushort(sw.s1);
  short s = sw.s0;
}

kernel void zero_tsdf(
        global short2 *voxels
)
{
  unsigned int iX = get_global_id(2);
  unsigned int iY = get_global_id(1);
  unsigned int iZ = get_global_id(0);
  unsigned int index = iX*(N_WIDTH*N_WIDTH) + iY*(N_WIDTH) + iZ;
  voxels[index] += (short2)(iX, 1);
}

kernel void zero_tsdf_2(
        global short2 *voxels
)
{
  unsigned int iZ = get_global_id(0);
  unsigned int iX = get_global_id(1);
  unsigned int index = iX*(N_WIDTH*N_WIDTH) + iZ;  
  for (unsigned int iY = 0; iY < N_WIDTH; iY++) {
    //unsigned int index = iX*(N_WIDTH*N_WIDTH) + iY*(N_WIDTH) + iZ;  
    voxels[index] += (short2)(iX, 1);
    index += N_WIDTH;
  }
}

kernel void flatrot_compute(
	global float4 *output,
	global const float4 *norm,
	float4 v0, float4 v1, float4 v2
)
{
  unsigned int index = get_global_id(0);
  if (norm[index].w == 0) { // Quit early if the weight is too low!
    output[index] = (float4)(0);
    return;  
  }
  float4 n = norm[index];
  float dx = dot(n, v0);
  float dy = dot(n, v1);
  float dz = dot(n, v2);
  
  float qz = 4*dz*dx*dx*dx - 4*dz*dz*dz * dx;
  float qx = dx*dx*dx*dx - 6*dx*dx*dz*dz + dz*dz*dz*dz;
  
  if (dy<0.3) output[index] = (float4)(qx, 0,qz, 1);  
  else        output[index] = (float4)(0,0,0,0);
}

kernel void float4_sum(
	global float4 *result,
	local float4 *scratch,
	global const float4 *input,
	const int length
)
{
  int global_index = get_global_id(0);
  float4 accum = (float4)(0);
  while (global_index < length) {
    accum += input[global_index];
    global_index += get_global_size(0);
  }
  int local_index = get_local_id(0);
  scratch[local_index] = accum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = get_local_size(0) / 2;
           offset > 0;
           offset = offset / 2) {
    if (local_index < offset) {
      float4 other = scratch[local_index + offset];
      float4 mine  = scratch[local_index];
      scratch[local_index] = mine + other;
    } 
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_index == 0) {
    result[get_group_id(0)] = scratch[0];
  }
}

kernel void lattice2_compute(
	global float4 *face_label,
	global float4 *qx2z2,
	global float4 *modelxyz,
	global const float4 *norm,
	global const float4 *xyz,
	float modulo,
	const float4 mm0, const float4 mm1, const float4 mm2
)
{
  unsigned int index = get_global_id(0);
    
  if (norm[index].w == 0) { // Quit early if the weight is too low!
   face_label[index] = (float4)(0,0,0,0);
   qx2z2[index] = (float4)(0,0,0,0);
   modelxyz[index] = (float4)(0,0,0,as_float((char4)1));
   return;
  }

  float4 mmat[3];
  mmat[0] = mm0;
  mmat[1] = mm1;
  mmat[2] = mm2;

  // Project the depth image
  float4 XYZ = matmul3(mmat, xyz[index]);

  // Project the normals
  float4 dxyz_ = norm[index]; dxyz_.w = 0;
  dxyz_ = matmul3(mmat, dxyz_);

  // Threshold the normals and pack it into one number as a label
  const float CLIM = 0.9486;
  float4 cxyz_ = step(dxyz_,(float4)(-CLIM)) + step(dxyz_,(float4)(CLIM)) - 1;
  XYZ.w = as_float(convert_uchar4(cxyz_+1));

  // Finally do the trig functions
  float2 qsin, qcos;
  qsin = sincos(XYZ.xz * modulo * TAU, &qcos);
  float2 qx = (float2)(qcos.x,qsin.x);
  float2 qz = (float2)(qcos.y,qsin.y);
  if (cxyz_.x == 0) qx = (float2)(0);
  if (cxyz_.z == 0) qz = (float2)(0);

  // output structure: 
  modelxyz[index] = XYZ;
  face_label[index] = -convert_float4(isnotequal(cxyz_,0));
  qx2z2[index] = (float4)(qx,qz);
}


kernel void gridinds_compute(
  global char4 *gridinds,
  global const float4 *modelxyz,
  const float xfix, const float zfix, 
  const float LW, const float LH,
  const float4 gridmin, const float4 gridmax
)
{
  unsigned int index = get_global_id(0);

  float4 xyzf = modelxyz[index];
  unsigned int asdf = as_uint(xyzf.w);
  float4 cxyz_;
  cxyz_.x = (asdf & 0xFF) - 1.0;
  cxyz_.y = ((asdf >>  8) & 0xFF) - 1.0;
  cxyz_.z = ((asdf >> 16) & 0xFF) - 1.0;

  float4 f1 = cxyz_ * 0.5;
  float4 fix = (float4)(xfix,0,zfix,0);  
  float4 mod = (float4)(LW,LH,LW,1);

  float4 occ = floor(-gridmin + (xyzf-fix)/mod + f1);
  float4 vac = occ - cxyz_;
  
  occ.w = cxyz_.x*4 + cxyz_.y*2 + cxyz_.z;
  vac.w = occ.w;

  if (occ.x < 0 || occ.y < 0 || occ.z < 0) occ.w = 0;
  if (occ.x >= (gridmax.x-gridmin.x) ||
      occ.y >= (gridmax.y-gridmin.y) ||
      occ.z >= (gridmax.z-gridmin.z)) occ.w = 0;
  if (vac.x < 0 || vac.y < 0 || vac.z < 0) vac.w = 0;
  if (vac.x >= (gridmax.x-gridmin.x-1) ||
      vac.y >= (gridmax.y-gridmin.y-1) ||
      vac.z >= (gridmax.z-gridmin.z-1)) vac.w = 0;
  
  gridinds[2*index+0] = convert_char4(occ);
  gridinds[2*index+1] = convert_char4(vac);
}
"""


def setup_kernel(mats=None):
    kernel_source = kernel_source_template
    
    global program
    program = cl.Program(context, kernel_source).build("-cl-mad-enable")
    
    # I have no explanation for this workaround. Presumably it's fixed in 
    # another version of pyopencl. Wtf. Getting the kernel like this
    # makes it go much faster when we __call__ it.
    def workaround(self):
        return self
    cl.Kernel.workaround = workaround
    program.integrate_tsdf = program.integrate_tsdf.workaround()
    program.zero_tsdf = program.zero_tsdf.workaround()
    program.zero_tsdf_2 = program.zero_tsdf_2.workaround()
setup_kernel()

print program.get_build_info(context.devices[0], cl.program_build_info.LOG)


def print_all():
  print_info(context.devices[0], cl.device_info)
  print_info(program, cl.program_info)
  print_info(program.normal_compute, cl.kernel_info)
  print_info(queue, cl.command_queue_info)


#print_all()
#raw_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*2)
#depth_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4)
fmt = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT)
depth_image = cl.Image(context, mf.READ_WRITE, fmt, (480,640))
#normals_buf = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
#voxels_buf = cl.Buffer(context, mf.READ_WRITE, 512*512*512*2*2)
voxels_buf = [cl.Buffer(context, mf.READ_WRITE, 64*512*512*2*2)
              for _ in range(8)]

#debug_buf     = cl.Buffer(context, mf.READ_WRITE, 480*640*4*4)
reduce_buf    = cl.Buffer(context, mf.READ_WRITE, 8*4*100)
reduce_scratch = cl.LocalMemory(64*8*4)

WIDTH = 640
HEIGHT = 480
N_WIDTH = 512

def load_raw(depth):
  assert depth.dtype == np.float32
  assert depth.shape == (HEIGHT, WIDTH)
  return cl.enqueue_write_buffer(queue, raw_buf, depth, is_blocking=False)

def get_normals():
  normals = np.empty((WIDTH*HEIGHT,4), 'f')
  cl.enqueue_read_buffer(queue, normals_buf, normals).wait()
  return normals.reshape(HEIGHT,WIDTH,4)

def compute_normals():
  kernel = program.normal_compute_ONE
  evt = kernel(queue, (height, width), None, normals_buf, xyz_buf,
               filt_buf, raw_buf, mask_buf,
               bounds, np.int32(0))  # offset unused (0)
  #import main
  #if main.WAIT_COMPUTE: evt.wait()
  return evt


def zero_tsdf():
    for i in range(len(voxels_buf)):
        evt = program.zero_tsdf(queue, (N_WIDTH,N_WIDTH,N_WIDTH/8),
                                None, voxels_buf[i])
        print 'i:', i
    evt.wait()
    return evt

def zero_tsdf_2():
    for i in range(len(voxels_buf)):
        evt = program.zero_tsdf_2(queue, (N_WIDTH,N_WIDTH/8),
                                None, voxels_buf[i])
        print 'i:', i
    #evt = program.zero_tsdf_2(queue, (N_WIDTH,N_WIDTH), None, voxels_buf)
    evt.wait()
    return evt

def reduce_flatrot():
  sums = np.empty((8,4),'f')  
  evt = program.float4_sum(queue, (64*8,), (64,), 
    reduce_buf, reduce_scratch, 
    qxdyqz_buf, np.int32(length))
  cl.enqueue_read_buffer(queue, reduce_buf, sums).wait()
  return sums.sum(0)
    
def reduce_lattice2():
  sums = np.empty((8,4),'f') 
  
  evt = program.float4_sum(queue, (64*8,), (64,), 
    reduce_buf, reduce_scratch, 
    qxqz_buf, np.int32(length))
  cl.enqueue_read_buffer(queue, reduce_buf, sums).wait()
  qxqz = sums.sum(0)  
  
  evt = program.float4_sum(queue, (64*8,), (64,), 
    reduce_buf, reduce_scratch, 
    face_buf, np.int32(length))
  cl.enqueue_read_buffer(queue, reduce_buf, sums).wait()
  cxcz = sums.sum(0)

  return cxcz,qxqz
