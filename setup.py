from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("rtmodel.fasticp",
                        ["rtmodel/fasticp.pyx"]),
             Extension("rtmodel.rangeimage_speed",
                        ["rtmodel/rangeimage_speed.pyx"]),
             ]

setup(name='Real-Time 3D Modeling (rtmodel)',
      version='0.01',
      packages=['rtmodel'],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules)
