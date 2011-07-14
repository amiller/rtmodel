import rangeimage
import pointmodel

DIST1 = 0.02
RATE1 = 0.15
DIST2 = 0.05
RATE_SMALL = 0.1
DIST_FINAL = 0.005
RATE_FINAL = 0.25
DIST_THRESH = 0.003
RATE_NEW_ANCHOR = 0.25 * RATE_FINAL
MAX_ANCHORS = 5


class RTModel(object):
    def __init__():
        self.anchors = []
        self.RT = np.eye(4, dtype='f')

    def add(self, rimg):
        rpoints = rimg.point_model()

        # Check that the frame is valid, more >= 500
        if rpoints.xyz.shape[0] < 500:
            print 'not enough points'
            return False

        # Align the anchors, starting with the front one
        n = 0
        whichanchor = 0
        if len(anchors):
            n, pnew = self.align(rimg, ptmodel, whichanchor, RATE1, DIST1, 4)
            if not n:
                for whichanchor in range(1, len(anchors)):
                    n, pnew = align(rimg, ptmodel, whichanchor, RATE2, DIST2, 6)

        # For the first frame - just add it, the rest of the frames
        # we'll need to decide to skip this frame or not
        if self.anchors == []:
            self.anchors.append(rimg)
        else:
            if whichanchor != 0:
                a = self.anchors.pop(whichanchor)
                self.anchors = [a] + self.anchors

            # Use the number of valid matches to decide if this is
            # a good new anchor to use
            av = self.anchors[0].nvalid
            rv = rimg.nvalid
            nthresh = int(RATE_NEW_ANCHOR * (av+rv))
            if (av < 1000 and rv > av) or \
               (av >= 1000 and rv >= 1000 and n < nthresh):
                
                if len(self.anchors) >= MAX_ANCHORS:
                    self.anchors.pop()

                # Prepend the new image
                self.anchors = [rimg] + self.anchors

                # TODO: now the display grid should be updated, if we're doing that
                
                # TODO: now would be a good time to export the model


    def align(self, rimg, ptmodel, rate, dist, miniters):

        rate = RATE_SMALL
        niters = 1
        pnew = ptmodel
        while 1:
            dist = 0.5*dist + 0.5*dist_final
            # Fix next line
            pnew, err, npairs, _ = fasticp.fast_icp(rimg, pnew,
                                                    int(640*480*rate), dist)
            lasterr = 0.75*lasterr + 0.25*err
            niters += 1
            if not (npairs > 25) and ((niters < miniters) or (err <= lasterr)):
                break

        rate = RATE_FINAL
        dist = DIST_FINAL
        if (npairs > 25):
            pnew, err, npairs, _ = fasticp.fast_icp(rimg, pnew,
                                                    int(640*480*rate), dist)

        if (npairs > 50) and (err < dist_thresh):
            return npairs, pnew
        
        return 0, ptmodel

    def align(rimg1, rimg2):
    pass
