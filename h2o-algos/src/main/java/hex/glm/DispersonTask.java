package hex.glm;

import water.Job;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.util.Log;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static hex.glm.DispersonTask.ComputeMaxSumSeriesTsk._ConstColNames.*;
import static hex.glm.DispersonTask.ComputeMaxSumSeriesTsk._InfoColNames.*;
import static org.apache.commons.math3.special.Gamma.gamma;
import static org.apache.commons.math3.special.Gamma.logGamma;

public class DispersonTask {
    public static final double LOG2 = Math.log(2);
    /***
     * Class to pre-calculate constants assocated with the following processes:
     * 1. maximum term index: jMaxConst (col 0) for 1<p<2, -jMaxConst for p>2
     * 2. constant term associated with z: zConst for 1<p<2, -zConst for p>2
     * 3. log likelihood: part2Const (col 1), same for all p
     * 4. log 1/y for 1<p<2 
     * 5. log 1/(Pi*y) for p>2
     * 5. dlogf/dphi firstOrderDerivConst, same for all p
     * 6. d2logf/dphi2 secondOrderDerivConst, same for all p
     * 
     * In addition, we also have maximum term with maximum index: logMaxConst, not part of constFrame.
     */
    public static class ComputeTweedieConstTsk extends MRTask<ComputeTweedieConstTsk> {
        double _variancePower;
        double _alpha;
        final Job _job;
        boolean _weightPresent;
        double _twoMinusP;
        double _oneOver2MinusP;
        double _alphaMinus1;
        double _log2Pi;
        double _logJMaxConst;
        double _oneMinusP;
        double _oneOver1MinusP;
        double _logKMaxConst;
        double _oneOverPi;
        
        public ComputeTweedieConstTsk(Job j, double vPower, Frame infoFrame) {
            _variancePower = vPower;
            _alpha = (2.0 - vPower) / (1.0 - vPower);
            _alphaMinus1 = _alpha-1;
            _job = j;
            _weightPresent = infoFrame.numCols() > 2;
            _twoMinusP = 2-vPower;
            _oneOver2MinusP = 1.0/_twoMinusP;
            _log2Pi = Math.log(2*Math.PI);
            _logJMaxConst = -Math.log(2*Math.PI)-0.5*Math.log(-_alpha);
            _logKMaxConst = 0.5*Math.log(_alpha);
            _oneMinusP = 1-_variancePower;
            _oneOver1MinusP = 1.0/_oneMinusP;
            _oneOverPi = 1.0/Math.PI;
        }
        
       public void map(Chunk[] chks, NewChunk[] constChks) {
           if (isCancelled() || _job != null && _job.stop_requested()) return;
            int chkLen = chks[0].len();
            for (int rowInd = 0; rowInd < chkLen; rowInd++) {
                // calculate jMaxConst
                calJMaxConst(chks, constChks, rowInd, 0);
                // calculate zConst
                calZConst(chks, constChks, rowInd, 1);
                // calculate part2Const
                calPart2Const(chks, constChks, rowInd, 2);
                // calculate part1Const, the 1/y
                calPart1Const(chks, constChks, rowInd, 3);
                // calculate partConst, 1/(PI*y)
                calPart1PIConst(chks, constChks, rowInd, 4);
                // calculate constants for derivatives
                calDerivConst(chks, constChks, rowInd, new int[] {5,6});
            }
       } 
       
       public void calZConst(Chunk[] chks, NewChunk[] constChks, int rowInd, int newChkColInd) {
            double response = chks[0].atd(rowInd);
            if (!Double.isNaN(response)) {
                if (response != 0) {
                    double val = Math.pow(response, -_alpha) * Math.pow(-_oneMinusP, _alpha) * _oneOver2MinusP;
                    if (_weightPresent)
                        val *= chks[2].atd(rowInd);
                    constChks[newChkColInd].addNum(val);
                } else 
                    constChks[newChkColInd].addNum(0);
            } else {
                constChks[newChkColInd].addNA();
            }
       }
       
       public void calDerivConst(Chunk[] chks, NewChunk[] constChks, int rowInd, int[] newChkColInd) {
            double response = chks[0].atd(rowInd);
            double mu = chks[1].atd(rowInd);
            double val;
            double weight = _weightPresent ? chks[2].atd(rowInd) : 1;
           if (!Double.isNaN(response) && !Double.isNaN(mu)) {
               val = -response*Math.pow(mu, _oneMinusP)*_oneOver1MinusP+Math.pow(mu, _twoMinusP)*_oneOver2MinusP;
                   val *= weight*weight;
                   constChks[newChkColInd[0]].addNum(val); // dll/dphi constant
                   val *= -2*weight;
                   constChks[newChkColInd[1]].addNum(val); // d2ll/dphi2 constant
           } else {
               constChks[newChkColInd[0]].addNA();
               constChks[newChkColInd[1]].addNA();
           }
       }
       
       public void calPart1Const(Chunk[] chks, NewChunk[] constChks, int rowInd, int newChkColInd) {
           double response = chks[0].atd(rowInd);
            if (!Double.isNaN(response) && response != 0) {
                constChks[newChkColInd].addNum(Math.log(1.0/response));
            } else {
                constChks[newChkColInd].addNA();
            }
       }

        public void calPart1PIConst(Chunk[] chks, NewChunk[] constChks, int rowInd, int newChkColInd) {
            double response = chks[0].atd(rowInd);
            if (!Double.isNaN(response) && response != 0) {
                constChks[newChkColInd].addNum(Math.log(_oneOverPi/response));
            } else {
                constChks[newChkColInd].addNA();
            }
        }
       
       public void calPart2Const(Chunk[] chks, NewChunk[] constChks, int rowInd, int newChkColInd) {
           double response = chks[0].atd(rowInd);
           double mu = chks[1].atd(rowInd);
           if (!Double.isNaN(response) && !Double.isNaN(mu)) {
               double val;
               if (response == 0) {
                   val = -Math.pow(mu, _twoMinusP)*_oneOver2MinusP;
               } else{
                   val = response*Math.pow(mu, _oneMinusP)*_oneOver1MinusP-
                           Math.pow(mu, _twoMinusP)*_oneOver2MinusP;
               }
               if (_weightPresent)
                   val *= chks[2].atd(rowInd);
               constChks[newChkColInd].addNum(val);
           } else {
               constChks[newChkColInd].addNA();
           }
       }
       
       public void calJMaxConst(Chunk[] chks, NewChunk[] constChks, int rowInd, int newChkColInd) {
           double response = chks[0].atd(rowInd);
           double mu = chks[1].atd(rowInd);
            if (!Double.isNaN(response) && !Double.isNaN(mu) && response != 0) {
                double val = _variancePower < 2 ? Math.pow(response, _twoMinusP)*_oneOver2MinusP : 
                        -Math.pow(response, _twoMinusP)*_oneOver2MinusP;
                if (_weightPresent)
                    val *= chks[2].atd(rowInd);
                constChks[newChkColInd].addNum(val);
           } else {
                constChks[newChkColInd].addNA();
           }
       }
    }

    /***
     * This class will compute the following for every row of the dataset:
     * 1. index of maximum magnitude of infinite series;
     * 2. log(z)
     * 3. W or V maximum
     * 5. dW or dV maximum
     * 6. d2W or d2V maximum
     * 7. KL or JL for W or V
     * 8. KU or JU for W or V
     * 9. KL or JL for dW or dV
     * 10. KU or JU for dW or dV
     * 11. KL or JL for d2W or d2V
     * 12. KU or JU for d2W or d2V
     * 13. log likelihood
     * 14. dlog likelihood / d phi
     * 15. d2log likelihood / d2 phi 
     */
    public static class ComputeMaxSumSeriesTsk extends MRTask<ComputeMaxSumSeriesTsk> {
        double _variancePower;
        double _dispersionParameter;
        double _alpha;
        Job _job;
        boolean _weightPresent;
        Frame _infoFrame;
        public enum _ConstColNames {JMaxConst, zConst, LogPart2Const, LogOneOverY, LogOneOverPiY, FirstOrderDerivConst,
            SecondOrderDerivConst};
        public enum _InfoColNames {MaxValIndex, LOGZ, LOGWVMax, LOGDWVMax, LOGD2WVMax, JkL, JkU, DjkL, DjkU, SumWV,
            SumDWV, SumD2WV, LL, DLL, D2LL};
        int _constColOffset;
        int _workColOffset;
        int _nWorkCols;
        double _oneOverPhiPower;
        double _oneMinusAlpha;
        double _oneOverPhiSquare;
        double _oneOverPhi3;
        double _logLL;
        double _dLogLL;
        double _d2LogLL;
        boolean _debugOn;
        double _oneOverDispersion;
        double _alphaMinus1TLogDispersion;
        double _alphaTimesPI;
        double _alphaMinus1OverPhi;
        double _alphaMinus1SquareOverPhiSquare;
        double _alphaMinus1OverPhiSquare;
        int _nWVs = 3;
        int _indexBound;
        double _logDispersionEpsilon;
        boolean[] _computationAccuracy; // set to false when upper bound exceeds _indexBound
        int _constantColumnNumber;

        public ComputeMaxSumSeriesTsk(Job j, TweedieMLDispersionOnly tdispersion, GLMModel.GLMParameters parms) {
            _variancePower = tdispersion._variancePower;
            _dispersionParameter = tdispersion._dispersionParameter;
            _alpha = (2.0-_variancePower)/(1.0-_variancePower);
            _job = j;
            _weightPresent = tdispersion._weightPresent;
            _infoFrame = tdispersion._infoFrame;
            _nWorkCols = tdispersion._nWorkingCol;
            _constantColumnNumber = tdispersion._constFrameNames.length;
            _constColOffset = _infoFrame.numCols()-_nWorkCols-tdispersion._constNCol;
            _workColOffset = _infoFrame.numCols()-_nWorkCols;
            _oneMinusAlpha = 1-_alpha;
            _oneOverPhiPower = 1.0/Math.pow(_dispersionParameter, _oneMinusAlpha);
            _oneOverPhiSquare = 1.0/(_dispersionParameter * _dispersionParameter);
            _oneOverPhi3 = _oneOverPhiSquare/_dispersionParameter;
            _debugOn = parms._debugTDispersionOnly;
            _oneOverDispersion = 1/_dispersionParameter;
            _alphaMinus1TLogDispersion = (_alpha-1)*Math.log(_dispersionParameter);
            _alphaTimesPI = _alpha*Math.PI;
            _indexBound = parms._max_series_index;
            _logDispersionEpsilon = Math.log(parms._tweedie_epsilon);
            _computationAccuracy = new boolean[_nWVs];
            _alphaMinus1OverPhi = (_alpha-1)/_dispersionParameter;
            _alphaMinus1SquareOverPhiSquare = _alphaMinus1OverPhi*_alphaMinus1OverPhi;
            _alphaMinus1OverPhiSquare = _alphaMinus1OverPhi/_dispersionParameter;
        }
        
        public void setIndices(Map<_ConstColNames, Integer> constColName2Ind, Map<_InfoColNames, Integer> infoColName2Ind) {
            DispersonTask.ComputeMaxSumSeriesTsk._ConstColNames[] constVal = 
                    DispersonTask.ComputeMaxSumSeriesTsk._ConstColNames.values();
            int offset = _weightPresent ? 3 : 2;
            for (int index = 0; index< _constantColumnNumber; index++)
                constColName2Ind.put(constVal[index], index+offset);
            
            DispersonTask.ComputeMaxSumSeriesTsk._InfoColNames[] infoC = 
                    DispersonTask.ComputeMaxSumSeriesTsk._InfoColNames.values();
            offset += constVal.length;
            int infoColLen = infoC.length;
            for (int index=0; index<infoColLen; index++) {
                infoColName2Ind.put(infoC[index], index+offset);
            }
        }

        public void map(Chunk[] chks) {
            int chunkID = chks[0].cidx();
            int chkLen = chks[0].len();
            int jKIndMax=0, jKL=0, jKU=0, djKL=0, djKU=0;
            double wvMax=0, dwvMax=0, d2wvMax=0, logZ=0, sumWVj=0, sumDWVj=0, sumD2WVj=0, oneOverSumWVj=0;
            _logLL = 0;  _dLogLL = 0; _d2LogLL = 0;
            double tempLL, tempDLL, tempD2LL;
            Map<_ConstColNames, Integer> constColName2Ind = new HashMap<>();
            Map<_InfoColNames, Integer> infoColName2Ind = new HashMap<>();
            setIndices(constColName2Ind, infoColName2Ind);
            for (int rInd = 0; rInd < chkLen; rInd++) {
                double response = chks[0].atd(rInd);
                if (response != 0) {
                    // calculate maximum index of series;
                    jKIndMax = findMaxTermIndex(chks, rInd, constColName2Ind.get(JMaxConst));
                    // calculate log(z)
                    logZ = calLogZ(chks, rInd, constColName2Ind.get(zConst));
                    // calculate maximum of W/V, dW/dV, d2W/dV2;
                    wvMax = calLogWVMax(chks, rInd, jKIndMax, constColName2Ind.get(zConst), _oneOverPhiPower);
                    dwvMax = wvMax + Math.log(jKIndMax);
                    d2wvMax = dwvMax + LOG2;
                    // locate jL/kL, jU/kU for W/V, dW/dV, d2W/dV2;
                    jKL = estimateLowerBound(jKIndMax, wvMax, logZ, 0, new EvalLogWVEnv());
                    jKU = estimateUpperBound(jKIndMax, wvMax, logZ, 0, new EvalLogWVEnv());
                    djKL = estimateLowerBound(jKIndMax, dwvMax, logZ, 1, new EvalLogDWVEnv());
                    djKU = estimateUpperBound(jKIndMax, dwvMax, logZ, 1, new EvalLogDWVEnv());
                    // sum the series W, dW, d2W
                    sumWVj = sumWV(jKL, jKU, wvMax, logZ, new EvalLogWVEnv()); // 1/y or 1/(Pi*y) cancelled in ratio. will ignore
                    oneOverSumWVj = 1.0 / sumWVj;
                    sumDWVj = sumWV(djKL, djKU, dwvMax, logZ, new EvalLogDWVEnv()) * _alphaMinus1OverPhi;
                    sumD2WVj = sumWV(djKL, djKU, d2wvMax, logZ, new EvalLogD2WVEnv()) * _alphaMinus1SquareOverPhiSquare
                    - sumDWVj*_oneOverDispersion;
                }
                // calculate loglikelihood, d ll/d phi, d2ll/dphi2 by doing the actual sum of the series
                tempLL = evalLogLikelihood(chks, rInd, sumWVj, constColName2Ind);
                if (!Double.isNaN(tempLL))
                    _logLL += tempLL;
                tempDLL = evalDlldPhi(chks, rInd, sumDWVj, oneOverSumWVj, constColName2Ind);
                if (!Double.isNaN(tempDLL))
                    _dLogLL += tempDLL;
                tempD2LL =  evalD2lldPhi2(chks, rInd, sumDWVj, sumD2WVj, oneOverSumWVj, constColName2Ind);
                if (!Double.isNaN(tempD2LL))
                    _d2LogLL += tempD2LL;
                if (_debugOn) {
                    if (response == 0) {
                        jKIndMax=0; jKL=0; jKU=0; djKL=0; djKU=0;
                        wvMax=0; dwvMax=0; d2wvMax=0; logZ=0; sumWVj=0; sumDWVj=0; sumD2WVj=0; oneOverSumWVj=0;
                    }
                    setDebugValues(rInd, jKIndMax, logZ, wvMax, dwvMax, d2wvMax, jKL, jKU, djKL, djKU, sumWVj, sumDWVj,
                            sumD2WVj, tempLL, tempDLL, tempD2LL, chks, infoColName2Ind);
                }
            }
        }
        
        public void setDebugValues(int rInd, int jkIndMax, double logZ, double wvMax, double dwvMax, double d2wvMax, 
                                   int jKL, int jKU, int djKL, int djKU, double sumWV, double sumDWV, double sumD2WV, 
                                   double ll, double dll, double d2ll, Chunk[] chks, 
                                   Map<_InfoColNames, Integer> infoColName2Ind) {
            chks[infoColName2Ind.get(MaxValIndex)].set(rInd, jkIndMax);
            chks[infoColName2Ind.get(LOGZ)].set(rInd, logZ);
            chks[infoColName2Ind.get(LOGWVMax)].set(rInd, wvMax);
            chks[infoColName2Ind.get(LOGDWVMax)].set(rInd, dwvMax);
            chks[infoColName2Ind.get(LOGD2WVMax)].set(rInd, d2wvMax);
            chks[infoColName2Ind.get(JkL)].set(rInd, jKL);
            chks[infoColName2Ind.get(JkU)].set(rInd, jKU);
            chks[infoColName2Ind.get(DjkL)].set(rInd, djKL);
            chks[infoColName2Ind.get(DjkU)].set(rInd, djKU);
            chks[infoColName2Ind.get(SumWV)].set(rInd, sumWV);
            chks[infoColName2Ind.get(SumDWV)].set(rInd, sumDWV);
            chks[infoColName2Ind.get(SumD2WV)].set(rInd, sumD2WV);
            chks[infoColName2Ind.get(LL)].set(rInd, ll);
            chks[infoColName2Ind.get(DLL)].set(rInd, dll);
            chks[infoColName2Ind.get(D2LL)].set(rInd, d2ll);    
        }
        
        @Override
        public void reduce(ComputeMaxSumSeriesTsk other) {
            this._logLL += other._logLL;
            this._dLogLL += other._dLogLL;
            this._d2LogLL += other._d2LogLL;
        }

        public int estimateLowerBound(int jOrkMax, double logWorVmax, double logZ, int wvIndex, CalWVdWVd2WV cVal) {
            if (jOrkMax == 1)   // small speedup
                return 1;
            double logWV1 = cVal.calculate(1, _alpha, logZ, logWorVmax, _variancePower);
            if ((logWV1 - logWorVmax) >= _logDispersionEpsilon)
                return 1;
            else {   // call recursive function
                int indexLow = 1;
                int indexHigh = jOrkMax;
                int indexMid = (int) Math.round(0.5*(indexLow+indexHigh));
                double logVal;
                while ((indexLow < indexHigh) && (indexHigh != indexMid) && (indexLow != indexMid)) {
                    logVal = cVal.calculate(indexMid, _alpha, logZ, logWorVmax, _variancePower);
                    if (logVal - logWorVmax < _logDispersionEpsilon)
                        indexLow = indexMid;
                    else
                        indexHigh = indexMid;
                    indexMid = (int) Math.round(0.5*(indexLow+indexHigh));
                }
                return indexMid;
            }
        }

        public int estimateUpperBound(int jOrkMax, double logWorVmax, double logZ, int wvIndex, CalWVdWVd2WV cVal) {
            double logWj = cVal.calculate(_indexBound, _alpha, logZ, logWorVmax, _variancePower);
            if ((logWj-logWorVmax) > _logDispersionEpsilon) {
                _computationAccuracy[wvIndex] = false;
                return _indexBound;
            } else {
                int indexLow = jOrkMax;
                int indexHigh = _indexBound;
                int indexMid = (int) Math.round(0.5*(indexLow+indexHigh));
                while ((indexLow < indexHigh) && (indexHigh != indexMid) && (indexLow != indexMid)) {
                    logWj = cVal.calculate(indexMid, _alpha, logZ, logWorVmax, _variancePower);
                    if (logWj-logWorVmax < _logDispersionEpsilon)
                        indexHigh = indexMid;
                    else
                        indexLow = indexMid;
                    indexMid = (int) Math.round(0.5*(indexLow+indexHigh)); 
                }
                return indexMid;
            }
        }
        

        double sumWV(int jkL, int jkU, double logWVMax, double logZ, CalWVdWVd2WV cCal) {
            if (_variancePower < 2)
                return (Math.exp(Math.log(IntStream.rangeClosed(jkL, jkU).mapToDouble(x->Math.exp(cCal.calculate(x, _alpha,
                        logZ, logWVMax, _variancePower)-logWVMax)).sum()) + logWVMax));
            else    // dealing with Vk, not using logWVMax because the sum can be slightly negative...
                return IntStream.rangeClosed(jkL, jkU).mapToDouble(x->Math.exp(cCal.calculate(x, 
                        _alpha, logZ, logWVMax, _variancePower))*Math.pow(-1, x)*Math.sin(-x*_alphaTimesPI)).sum();
        }

        public int findMaxTermIndex(Chunk[] chks, int rowInd, int colInd) {
            return (int) Math.max(1, Math.ceil(chks[colInd].atd(rowInd)*_oneOverDispersion));
        }

        public double calLogZ(Chunk[] chks, int rInd, int zConstCol) {
            if (_variancePower < 2)
                return Math.log(chks[zConstCol].atd(rInd))-_alphaMinus1TLogDispersion;
            else
                return Math.log(-chks[zConstCol].atd(rInd))-_alphaMinus1TLogDispersion;
        }

        public double calLogWVMax(Chunk[] chks, int rowInd, int indexMax, int zConstInd, double oneOverPhiPower) {
            if (_variancePower < 2) {    //  1 < p < 2
                return indexMax*Math.log(chks[zConstInd].atd(rowInd)*oneOverPhiPower)-Math.log(gamma(1+indexMax))-
                        Math.log(gamma(-_alpha*indexMax));
            } else { //p > 2 
                return indexMax*Math.log(-chks[zConstInd].atd(rowInd)*oneOverPhiPower)+
                        Math.log(gamma(1+_alpha*indexMax))-Math.log(gamma(1+indexMax));
            }
        }
        
        public double evalDlldPhi(Chunk[] chks, int rowInd, double sumDWVj, double oneOverSumWVj, 
                                  Map<_ConstColNames, Integer> constColName2Ind) {
            double response = chks[0].atd(rowInd);
            if (response == 0) 
                return chks[constColName2Ind.get(FirstOrderDerivConst)].atd(rowInd)*_oneOverPhiSquare;
            else if (!Double.isNaN(response))
                return chks[constColName2Ind.get(FirstOrderDerivConst)].atd(rowInd)*_oneOverPhiSquare+
                        sumDWVj*oneOverSumWVj;
            else
                return 0.0;
        }
        
        public double evalD2lldPhi2(Chunk[] chks, int rowInd, double sumDWVj, double sumD2WVj, double oneOverSumWVj, 
                                    Map<_ConstColNames, Integer> constColName2Ind) {
            double response = chks[0].atd(rowInd);
            if (response == 0) {
                return chks[constColName2Ind.get(SecondOrderDerivConst)].atd(rowInd) * _oneOverPhi3;
            } else if (!Double.isNaN(response)) {
                return chks[constColName2Ind.get(SecondOrderDerivConst)].atd(rowInd) * _oneOverPhi3 + 
                        sumD2WVj * oneOverSumWVj - sumDWVj * sumDWVj * oneOverSumWVj * oneOverSumWVj;
            } else {
                return 0.0;
            }
        }

        public double evalLogLikelihood(Chunk[] chks, int rowInd, double sumWV, 
                                        Map<_ConstColNames, Integer> constColName2Ind) {
            double response = chks[0].atd(rowInd);
            double logPart2 = _oneOverDispersion*chks[constColName2Ind.get(LogPart2Const)].atd(rowInd);
            if (!Double.isNaN(response)) {
                if (response == 0.0) {
                    return logPart2;
                } else {
                    if (_variancePower < 2)
                        return Math.log(sumWV/chks[0].atd(rowInd))+logPart2;
                    else 
                        return Math.log(sumWV/(Math.PI*chks[0].atd(rowInd)))+logPart2;
                }
            } else {
                return 0.0;
            }
        }

        /***
         * This interface is used to calculate one item of the series in log.
         */
        public interface CalWVdWVd2WV {
            public double calculate(int jOrk, double alpha, double logZ, double funcMax, double varianceP);
        }

        public static class EvalLogWVEnv implements CalWVdWVd2WV {
            @Override
            public double calculate(int jOrk, double alpha, double logZ, double funcMax, double varianceP) {
                if (varianceP < 2) {
                    return jOrk * logZ - logGamma(1+jOrk) - logGamma(-alpha * jOrk);
                } else {
                    return jOrk * logZ + logGamma(1 + alpha * jOrk) - logGamma(1+jOrk);
                }
            }
        }
        
        public static class EvalLogDWVEnv implements CalWVdWVd2WV {

            @Override
            public double calculate(int jOrk, double alpha, double logZ, double funcMax, double varianceP) {
                return (new EvalLogWVEnv()).calculate(jOrk, alpha, logZ, funcMax, varianceP)+Math.log(jOrk);
            }
        }
        
        public static class EvalLogD2WVEnv implements CalWVdWVd2WV {

            @Override
            public double calculate(int jOrk, double alpha, double logZ, double funcMax, double varianceP) {
                return (new EvalLogDWVEnv()).calculate(jOrk, alpha, logZ, funcMax, varianceP) + LOG2;
            }
        }
    }
}
