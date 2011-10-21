import rangeimage
import pointmodel
import fasticp
import numpy as np

"""
DIST1 = 0.02
RATE1 = 0.15
DIST2 = 0.05
RATE2 = 0.25
RATE_SMALL = 0.1
DIST_FINAL = 0.005
RATE_FINAL = 0.25
DIST_THRESH = 0.003
RATE_NEW_ANCHOR = 0.25 * RATE_FINAL
MAX_ANCHORS = 5
"""
DIST1 = 0.05
RATE1 = 0.15
DIST2 = 0.05
RATE2 = 0.25
RATE_SMALL = 0.1
DIST_FINAL = 0.0005
RATE_FINAL = 0.25
DIST_THRESH = 0.0001
RATE_NEW_ANCHOR = 0.25 * RATE_FINAL
MAX_ANCHORS = 5


class RTModel(object):
    def __init__(self):
        self.anchors = []
        self.RT = np.eye(4, dtype='f')

    def add(self, rimg):
        pnew = rimg.point_model()
        rimg.nvalid = pnew.xyz.shape[0]        

        # Check that the frame is valid, more >= 500
        if rimg.nvalid < 500:
            print 'not enough points'
            return False

        # Align the anchors, starting with the front one
        n = 0
        whichanchor = 0
        if len(self.anchors):
            n, pnew = self.align(self.anchors[whichanchor], pnew, RATE1, DIST1, 4)
            if n:
                print("Matched against anchor 0 with %d matches" % n)

            # If we don't match to the front, then we need to try all the others
            if not n:
                for whichanchor in range(1, len(self.anchors)):
                    pnew.RT = self.anchors[whichanchor].camera.RT
                    n, pnew = self.align(self.anchors[whichanchor], pnew, RATE2, DIST2, 6)
                    if n:
                        print('OK with anchor %d' % whichanchor)
                        break
            if not n:
                return False

        # For the first frame - just add it, the rest of the frames
        # we'll need to decide to skip this frame or not
        def prepend(rimg):
            rimg.camera.RT = pnew.RT
            rimg.pm = rimg.point_model(True)
            self.anchors.insert(0, rimg)
            
        if self.anchors == []:
            prepend(rimg)
            print("Anchors was empty, adding the current image to start")
        else:
            if whichanchor != 0:
                a = self.anchors.pop(whichanchor)
                self.anchors.insert(0, a)
                print("Moving anchor %d to the front" % whichanchor)

            # Use the number of valid matches to decide if this is
            # a good new anchor to use
            av = self.anchors[0].nvalid
            rv = rimg.nvalid
            nthresh = int(RATE_NEW_ANCHOR * (av+rv))
            print('av:', av, 'rv:', rv, 'n:', n, 'nthresh:', nthresh)
            if (av < 1000 and rv > av) or \
               (av >= 1000 and rv >= 1000 and n < nthresh):
                print("Critera satisfied, adding a new anchor")
                
                if len(self.anchors) >= MAX_ANCHORS:
                    print("Too many anchors, popping one")
                    self.anchors.pop()

                # Prepend the new image
                prepend(rimg)

                # TODO: now the display grid should be updated, if we're doing that
                
                # TODO: now would be a good time to export the model
        return pnew

    def align(self, rimg, pnew, rate, dist, miniters):        
        rate = RATE_SMALL
        niters = 1
        lasterr = 0.
        while 1:
            dist = 0.5*dist + 0.5*DIST_FINAL
            try:
                pnew, err, npairs, _ = fasticp.fast_icp(rimg, pnew, rate, dist)
            except np.linalg.LinAlgError:
                print 'singular matrix'
                return 0, pnew

            lasterr = 0.75*lasterr + 0.25*err
            niters += 1
            if (npairs > 50) and ((niters < miniters) or (err <= lasterr)):
                continue
            else:
                break

        rate = RATE_FINAL
        dist = DIST_FINAL
        if (npairs > 25):
            try:
                pnew, err, npairs, _ = fasticp.fast_icp(rimg, pnew, rate, dist)
            except np.linalg.LinAlgError:
                print 'singular matrix'
                return 0, pnew

        if (npairs > 50) and (err < DIST_THRESH):
            print('err:', err)
            return npairs, pnew
        
        return 0, pnew
