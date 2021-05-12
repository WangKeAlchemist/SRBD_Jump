def fifthOrderPolynomialInterpolation(tDuration, tCurrent, startPos, startVel, startAcc, endPos, endVel, endAcc):
    coeffs = [0, 0, 0, 0, 0, 0]
    coeffs[0] = -0.5 * (
                12 * startPos - 12 * endPos + 6 * startVel * tDuration + 6 * endVel * tDuration + startAcc * pow(
            tDuration, 2) - endAcc * pow(tDuration, 2)) / pow(tDuration, 5)
    coeffs[1] = 0.5 * (
                30 * startPos - 30 * endPos + 16 * startVel * tDuration + 14 * endVel * tDuration + 3 * startAcc * pow(
            tDuration, 2) - 2 * endAcc * pow(tDuration, 2)) / pow(tDuration, 4)
    coeffs[2] = -0.5 * (
                20 * startPos - 20 * endPos + 12 * startVel * tDuration + 8 * endVel * tDuration + 3 * startAcc * pow(
            tDuration, 2) - endAcc * pow(tDuration, 2)) / pow(tDuration, 3)
    coeffs[3] = 0.5 * startAcc
    coeffs[4] = startVel
    coeffs[5] = startPos
    position = coeffs[0] * pow(tCurrent, 5) + coeffs[1] * pow(tCurrent, 4) + coeffs[2] * pow(tCurrent, 3) + coeffs[
        3] * pow(tCurrent, 2) + coeffs[4] * pow(tCurrent, 1) + coeffs[5]
    velocity = 5 * coeffs[0] * pow(tCurrent, 4) + 4 * coeffs[1] * pow(tCurrent, 3) + 3 * coeffs[2] * pow(tCurrent,
                                                                                                         2) + 2 * \
               coeffs[3] * pow(tCurrent, 1) + coeffs[4]
    acceleration = 20 * coeffs[0] * pow(tCurrent, 3) + 12 * coeffs[1] * pow(tCurrent, 2) + 6 * coeffs[2] * pow(tCurrent,
                                                                                                               1) + 2 * \
                   coeffs[3]
    return position, velocity, acceleration