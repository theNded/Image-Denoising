//
//  cvBM3D.cpp
//  testCV
//
//  Created by Neo on 14-5-30.
//  Copyright (c) 2014年 Neo. All rights reserved.
//

#include "core.h"
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

#define _TEST

static const double TAU_match = 400, HARD_filter = 2.0 * 20;
static const int    MAX_VEC_LEN = 35;

typedef struct CvBlock_t {
    CvMat *mat;
    int x;
    int y;
    double dist;
    double w;
    bool operator < (const CvBlock_t & a) const {
        return dist < a.dist;
    }
} CvBlock;

typedef struct CvDCT_t {
    CvMat *mat;
    bool calculated;
} CvDCT;

static CvBlock cvBlock(CvMat *mat, int x, int y, double dist) {
    CvBlock blk = {0};
    blk.mat = mat, blk.x = x, blk.y = y, blk.dist = dist;
    blk.w = 0;
    return blk;
}

static double cvDist(CvMat *a, CvMat *b) {
    double *ptra = a->data.db, *ptrb = b->data.db, vala, valb, dist = 0;
    int step = a->step / sizeof(double);
    for (int x = 0; x < a->width; x ++) {
        for (int y = 0; y < a->height; y ++) {
            vala = ptra[y * step + x];
            valb = ptrb[y * step + x];
            dist += (vala - valb) * (vala - valb);
        }
    }
    return dist / (a->width * a->height);
}

static void cvHardFilter(CvMat *a, double threshold) {
    double *ptr = a->data.db;
    int step = a->step / sizeof(double);
    for (int x = 0; x < a->width; x ++) {
        for (int y = 0; y < a->height; y ++) {
            if (abs(ptr[y * step + x]) < threshold) {
                ptr[y * step + x] = 0;
            }
        }
    }
}

IplImage * cvBM3D(IplImage *in, int windowSearch, int windowBlock, double sigma) {
    int windowSearch_2 = windowSearch >> 1,
        imageSize = in->height * in->width;
    
    out = cvCreateImage(cvGetSize(in), in->depth, 1);
    
    CvMat
    *ctr = cvCreateMat(windowBlock, windowBlock, CV_64F),
    *cmp = cvCreateMat(windowBlock, windowBlock, CV_64F),
    *iDCTBuf = cvCreateMat(windowBlock, windowBlock, CV_64F),
    *buf1dSrc, *buf1dDst;
    
    /* Initialize Buffer */

    CvDCT *dctBuf = new CvDCT[in->height * in->width],
    *dctBufPtrCtr, *dctBufPtrCmp;
    double *numerator    = new double[in->height * in->width];
    double *denominator  = new double[in->height * in->width];

    for (int i = 0; i < imageSize; i ++) {
        dctBuf[i].mat = cvCreateMat(windowBlock, windowBlock, CV_64F);
        cvSetZero(dctBuf[i].mat);
        dctBuf[i].calculated = false;
        numerator[i] = 0;
        denominator[i] = 0;
    }
    
    vector <CvBlock> blockVec;
    double dist;
    int loopWidth = in->width - windowBlock + 1,
        loopHeight = in->height - windowBlock + 1,
        cmpX, cmpY;
    
    for (int x = 0; x < loopWidth; x += 3) {
        for (int y = 0; y < loopHeight; y += 3) {
            /* Get Center Matrix for Efficiency */
            cvSetImageROI(in, cvRect(x, y, windowBlock, windowBlock));
            cvConvert(in, ctr);
            cvResetImageROI(in);

            /* 2D transform here for efficiency */
            dctBufPtrCtr = &dctBuf[y * in->width + x];
            
            if (! dctBufPtrCtr->calculated) {
                cvDCT(ctr, dctBufPtrCtr->mat, CV_DXT_FORWARD);
                dctBufPtrCtr->calculated = true;
            }
            
            blockVec.clear();
            /* Iterate & Calculate each Similar Blocks */
            for (int sX = - windowSearch_2; sX < windowSearch_2; sX ++) {
                for (int sY = - windowSearch_2; sY < windowSearch_2; sY ++) {
                    cmpX = x + sX, cmpY = y + sY;
                    
                    /* Check whether inside the image */
                    if (cmpX < 0 || cmpY < 0 || cmpX >= loopWidth || cmpY >= loopHeight)
                        continue;
                    
                    /* Get Near Matrix for Calculation */
                    cvSetImageROI(in, cvRect(cmpX, cmpY, windowBlock, windowBlock));
                    cvConvert(in, cmp);
                    cvResetImageROI(in);
                    
                    /* Calculate Similarity */
                    /* 2D transform here for efficiency */

                    dctBufPtrCmp = &dctBuf[cmpY * in->width + cmpX];
                    if (! dctBufPtrCmp->calculated) {
                        cvDCT(cmp, dctBufPtrCmp->mat, CV_DXT_FORWARD);
                        dctBufPtrCmp->calculated = true;
                    }
                    
                    dist = cvDist(dctBufPtrCtr->mat, dctBufPtrCmp->mat);

                    if (dist < TAU_match) {
                        blockVec.push_back(cvBlock(dctBufPtrCmp->mat, cmpX, cmpY, dist));
                    }
                }
            }
            
            /* 1D transform and hardThreshold */
            int step = blockVec[0].mat->step / sizeof(double),
            size = (int) blockVec.size(), pad1 = 0;

            if (size > MAX_VEC_LEN) {
                sort(blockVec.begin(), blockVec.end());
                size = MAX_VEC_LEN;
            }
            if (size & 1) {
                pad1 = 1;
            }
            // cout << size << endl;
            buf1dSrc = cvCreateMat(1, size + pad1, CV_64F);
            buf1dDst = cvCreateMat(1, size + pad1, CV_64F);
            for (int bX = 0; bX < windowBlock; bX ++) {
                for (int bY = 0; bY < windowBlock; bY ++) {
                   
                    /* Put a 1d vector of point in DCT field into a buf vector */
                    double *ptrBuf1dRd, *ptrBuf1dWr;

                    ptrBuf1dRd = buf1dSrc->data.db;
                    for (int iter = 0; iter < size; iter ++) {
                        *ptrBuf1dRd = blockVec[iter].mat->data.db[bY * step + bX];
                        ptrBuf1dRd ++;
                    }
                    if (pad1) {
                        *ptrBuf1dRd = 0;
                    }
                    
                    /* 1D transform */
                    cvDCT(buf1dSrc, buf1dDst, CV_DXT_FORWARD);
                    cvHardFilter(buf1dDst, HARD_filter);
                    cvDCT(buf1dDst, buf1dSrc, CV_DXT_INVERSE);
                    
                    /* Add to non-zero weight */
                    ptrBuf1dRd = buf1dDst->data.db;
                    ptrBuf1dWr = buf1dSrc->data.db;
                    for (int iter = 0; iter < size; iter ++) {
                        if (abs(*ptrBuf1dRd) > 1e-5)
                            blockVec[iter].w += 1;
                        blockVec[iter].mat->data.db[bY * step + bX] = *ptrBuf1dWr;
                        ptrBuf1dRd ++, ptrBuf1dWr ++;
                    }
                }
            }
            cvReleaseMat(&buf1dSrc);
            cvReleaseMat(&buf1dDst);
            
            /*  2D inverse transform */
            for (int iter = 0; iter < size; iter ++) {
                if (abs(blockVec[iter].w) < 1e-5) {
                    blockVec[iter].w = 1;
                } else {
                    blockVec[iter].w = 1 / blockVec[iter].w;
                }
                
                cvDCT(blockVec[iter].mat, iDCTBuf, CV_DXT_INVERSE);
                
                for (int bX = 0; bX < windowBlock; bX ++) {
                    for (int bY = 0; bY < windowBlock; bY ++) {
                        int index = (blockVec[iter].y + bY) * in->width + (blockVec[iter].x + bX);
                        numerator[index]
                        += blockVec[iter].w * round(iDCTBuf->data.db[bY * step + bX]);
                        denominator[index]
                        += blockVec[iter].w;
                    }
                }
            }
        }
    }
    
    /* Get Mixed Picture */
    uchar *outPtr = (uchar *)out->imageData;
    int step = out->widthStep;
    for (int x = 0; x < out->width; x ++) {
        for (int y = 0; y < out->height; y ++) {
            outPtr[y * step + x]
            = (uchar) (numerator[y * in->width + x] / denominator[y * in->width + x]);
        }
    }
    cvReleaseMat(&ctr);
    cvReleaseMat(&cmp);
    cvReleaseMat(&iDCTBuf);
    for (int i = 0; i < imageSize; i ++) {
        cvReleaseMat(&dctBuf[i].mat);
    }
    return out;
}