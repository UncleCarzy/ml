import numpy as np
import random


class optStruct(object):
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.asmatrix(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.asmatrix(np.zeros((self.m, 2)))


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T *
                (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    yEi = oS.labelMat[i] * Ei
    if ((yEi < - oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((yEi > oS.tol) and (oS.alphas[i] > 0.0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0.0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0.0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if abs(L - H) < oS.tol:
            # print("L == H")
            return 0
        eta = oS.X[i, :] * oS.X[i, :].T + oS.X[j, :] * \
            oS.X[j, :].T - 2.0 * oS.X[i, :] * oS.X[j, :].T

        if eta <= 0.0:
            # print("eta >= 0")
            return 0

        oS.alphas[j] += oS.labelMat[j] * (Ei-Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)

        if abs(oS.alphas[j] - alphaJold) < oS.tol:
            # print("j not moving enough")
            return 0

        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * \
            (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
            oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
                (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T

        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
            oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * \
                (oS.alphas[j] - alphaJold) * oS.X[j, :]*oS.X[j, :].T

        if 0 < oS.alphas[i] and oS.C > oS.alphas[i]:
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.matrix(dataMatIn), np.matrix(
        classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairChanged = 0
    while (iter < maxIter) and ((alphaPairChanged > 0) or entireSet):
        alphaPairChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairChanged += innerL(i, oS)
            # print("fullSet, iter: %d \t i: %d, pairs changed %d" %
            #       (iter, i, alphaPairChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairChanged += innerL(i, oS)
                # print("non-bound, iter: %d \t i: %d, pairs changed %d" %
                #   (iter, i, alphaPairChanged))
            iter += 1

        if entireSet:
            entireSet = False
        elif alphaPairChanged == 0:
            entireSet = True

        # print("iteration number: %d" % iter)
    print("total iteration number: %d" % iter)
    return oS.b, oS.alphas
