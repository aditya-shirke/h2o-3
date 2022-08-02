package hex.glm;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;
import water.fvec.Vec;
import water.runner.CloudSize;
import water.runner.H2ORunner;
import water.util.Log;

import java.util.Arrays;
import java.util.stream.IntStream;

import static hex.glm.GLMModel.GLMParameters.DispersionMethod.pearson;
import static hex.glm.GLMModel.GLMParameters.Family.tweedie;
import static org.apache.commons.math3.special.Gamma.gamma;
import static org.apache.commons.math3.special.Gamma.logGamma;

@RunWith(H2ORunner.class)
@CloudSize(1)
public class GLMTweedieDispersionOnlyTest extends TestUtil {
    
    @Test
    public void testTweedieDispersionEstimation() {
        Scope.enter();
        try {
            Frame train = parseAndTrackTestFile("smalldata/glm_test/tweedie_1p8Power_2Dispersion_5Col_10KRows.csv");
            GLMModel.GLMParameters params = new GLMModel.GLMParameters();
            params._tweedie_variance_power=1.8;
            params._family = tweedie;
            params._fix_tweedie_variance_power = true;
            params._debugTDispersionOnly = true;
            params._dispersion_factor_method = GLMModel.GLMParameters.DispersionMethod.ml;
            params._compute_p_values = true;
            params._lambda = new double[]{0.0};
            params._response_column = "resp";
            params._train = train._key;
            params._init_dispersion_parameter = 2.0;
            GLMModel glmML = new GLM(params).trainModel().get();
            Scope.track_generic(glmML);
            double trueDispersion = 2;
        } finally {
            Scope.exit();
        }
    }
    /***
     * This test is written to make sure working columns generated is correct for variance power p, 1<p<2 and p>2
     */
    @Test
    public void testInfoColGeneration() {
        Scope.enter();
        try {
            final Frame trainL2 = parseAndTrackTestFile("smalldata/glm_test/tweedie_5Cols_500Rows_power_1p5_phi_0p5.csv");
            GLMModel.GLMParameters params = new GLMModel.GLMParameters();
            params._tweedie_variance_power = 1.5;
            params._response_column = "resp";
            params._debugTDispersionOnly = true;
            assertCorrectInfoColGeneration(trainL2, params, 1e-6);
            // test for p > 2
            final Frame trainExceed2 = parseAndTrackTestFile("smalldata/glm_test/tweedie_5Cols_500Rows_power_2p1_phi_0p5.csv");
            params._tweedie_variance_power = 2.1;
            assertCorrectInfoColGeneration(trainExceed2, params, 1e-2);
        } finally {
            Scope.exit();
        }
    }

    @Test
    public void testInfoColGenerationWithWeight() {
        Scope.enter();
        try {
            final Frame trainL2 = parseAndTrackTestFile("smalldata/glm_test/tweedie_5Cols_500Rows_power_1p5_phi_0p5.csv");
            assertCorrectInfoColWithWeight(trainL2);

            final Frame trainE2 = parseAndTrackTestFile("smalldata/glm_test/tweedie_5Cols_500Rows_power_2p1_phi_0p5.csv");
            assertCorrectInfoColWithWeight(trainE2);
        } finally {
            Scope.exit();
        }
    }
    
    public void assertCorrectInfoColWithWeight(Frame train) {
        GLMModel.GLMParameters params = new GLMModel.GLMParameters();
        params._tweedie_variance_power = 1.5;
        params._response_column = "resp";
        params._debugTDispersionOnly = true;
        params._train = train._key;
        params._compute_p_values = true;
        params._dispersion_factor_method = pearson;
        params._family = tweedie;
        params._lambda = new double[]{0.0};
        GLMModel glmML = new GLM(params).trainModel().get();
        Scope.track_generic(glmML);
        TweedieMLDispersionOnly mlDisp = new TweedieMLDispersionOnly(train, params, glmML, glmML._output._job);
        DispersonTask.ComputeMaxSumSeriesTsk computeTask = new DispersonTask.ComputeMaxSumSeriesTsk(glmML._output._job,
                mlDisp, params);
        computeTask.doAll(mlDisp._infoFrame);   // generated info columns
        DKV.put(mlDisp._infoFrame);

        Vec vecOf1 = Vec.makeCon(1.0, train.numRows());
        train.add("weight", vecOf1);
        DKV.put(train);
        params._weights_column = "weight";
        GLMModel glmMLW = new GLM(params).trainModel().get();
        Scope.track_generic(glmMLW);
        TweedieMLDispersionOnly mlDispW = new TweedieMLDispersionOnly(train, params, glmMLW, glmMLW._output._job);
        DispersonTask.ComputeMaxSumSeriesTsk computeTaskW = new DispersonTask.ComputeMaxSumSeriesTsk(glmMLW._output._job,
                mlDispW, params);
        computeTaskW.doAll(mlDispW._infoFrame);   // generated info columns
        DKV.put(mlDispW._infoFrame);
        for (int cInd=2; cInd<mlDisp._infoFrame.numCols(); cInd++) {
            TestUtil.assertVecEquals(mlDisp._infoFrame.vec(cInd), mlDispW._infoFrame.vec(cInd+1), 1e-7);
        }
        Scope.track(mlDisp._infoFrame);
        Scope.track(mlDispW._infoFrame);
        mlDisp.cleanUp();
        mlDispW.cleanUp();
    }

    /***
     * This test is written to make sure the constant columns generation is correct for variance power p, 1 < p < 2 and
     * p > 2.
     */
    @Test
    public void testConstColumnGeneration() {
        Scope.enter();
        try {
            Frame train = parseAndTrackTestFile("smalldata/glm_test/tweedie_5Cols_500Rows_power_1p2_phi_0p5.csv");
            GLMModel.GLMParameters params = new GLMModel.GLMParameters();
            params._tweedie_variance_power = 1.2;
            params._response_column = "resp";
            assertConstColGeneration(train, params);
            // test for p > 2
            final Frame trainExceed2 = parseAndTrackTestFile("smalldata/glm_test/tweedie_5Cols_500Rows_power_2p1_phi_0p5.csv");
            params._tweedie_variance_power = 2.1;
            assertConstColGeneration(trainExceed2, params);
            
            // test with weight columns of 1.0.  Should provide the same result, 1<p<2
            Vec vecOf1 = Vec.makeCon(1.0, train.numRows());
            train.add("weight", vecOf1);
            DKV.put(train);
            params._weights_column = "weight";
            assertConstColGeneration(train, params);
            // test with weight and p>2
            trainExceed2.add("weight", vecOf1);
            DKV.put(trainExceed2);
            params._weights_column = "weight";
            assertConstColGeneration(train, params);
        } finally {
            Scope.exit();
        }
    }

    public void assertCorrectInfoColGeneration(Frame train, GLMModel.GLMParameters params, double tot) {
        params._train = train._key;
        params._compute_p_values = true;
        params._dispersion_factor_method = pearson;
        params._family = tweedie;
        params._lambda = new double[]{0.0};
        GLMModel glmML = new GLM(params).trainModel().get();
        Scope.track_generic(glmML);
        TweedieMLDispersionOnly mlDisp = new TweedieMLDispersionOnly(train, params, glmML, glmML._output._job);
        DispersonTask.ComputeMaxSumSeriesTsk computeTask = new DispersonTask.ComputeMaxSumSeriesTsk(glmML._output._job,
                mlDisp, params);
        computeTask.doAll(mlDisp._infoFrame);   // generated info columns
        DKV.put(mlDisp._infoFrame);
        compareInfoColFrame(computeTask, params, mlDisp, tot);
        mlDisp.cleanUp();
    }

    public void compareInfoColFrame(DispersonTask.ComputeMaxSumSeriesTsk computeTsk,  GLMModel.GLMParameters parms,
                                    TweedieMLDispersionOnly mlDisp, double tot) {
        Frame infoColFrame = mlDisp._infoFrame;
        Scope.track(infoColFrame);
        String[] infoColNames = mlDisp._workFrameNames;
        int infoColNum = infoColNames.length;
        int numRows = (int) infoColFrame.numRows();
        double[] manualInfoColsRow = new double[infoColNum];
        double[] infoColsRow = new double[infoColNum];
        double loglikelihood = 0;
        double dLoglikelihood = 0;
        double d2Loglikelihood = 0;
        int lastInd = infoColNum-1;
        
        int offset = mlDisp._weightPresent ? 3+mlDisp._constFrameNames.length : 2+mlDisp._constFrameNames.length;
        for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
            extractFrame2Array(infoColsRow, infoColFrame, offset, rowIndex);
            manuallyGenerateInfoCols(manualInfoColsRow, mlDisp, parms, rowIndex, infoColsRow);
            loglikelihood += manualInfoColsRow[lastInd-2];
            dLoglikelihood += manualInfoColsRow[lastInd-1];
            d2Loglikelihood += manualInfoColsRow[lastInd];
            Log.info("row index "+rowIndex);
            Assert.assertTrue(TestUtil.equalTwoArrays(manualInfoColsRow, infoColsRow, tot));
        }
        // check loglikelihood, dloglikelihood and d2loglikelihood sums
        Assert.assertTrue(Math.abs(Math.round(loglikelihood)-Math.round(computeTsk._logLL)) < tot);
        Assert.assertTrue(Math.abs(Math.round(dLoglikelihood)-Math.round(computeTsk._dLogLL)) < tot);
        Assert.assertTrue(Math.abs(Math.round(d2Loglikelihood)-Math.round(computeTsk._d2LogLL)) < tot);
    }
    
    public void manuallyGenerateInfoCols(double[] manualInfoCols, TweedieMLDispersionOnly mlDisp,
                                         GLMModel.GLMParameters params, int rowInd, double[] infoColRow) {
        Frame respMu = mlDisp._infoFrame;
        Scope.track(respMu);
        double resp = respMu.vec(0).at(rowInd);
        double mu = respMu.vec(1).at(rowInd);
        double weight = mlDisp._weightPresent ? respMu.vec(2).at(rowInd) : 1;
        double p = params._tweedie_variance_power;    
        double phi = mlDisp._dispersionParameter;
        double alpha = (2.0-p)/(1.0-p);
        if (!Double.isNaN(resp)) {
            // generate jKMaxIndex
            double jkMax = resp == 0 ? 0 : Math.max(1, Math.ceil(p < 2 ? weight*Math.pow(resp, 2-p)/((2-p)*phi) : 
                    weight*Math.pow(resp, 2-p)/((p-2)*phi)));
            manualInfoCols[0] = jkMax;
            // generate logZ
            double logZ = resp == 0 ? 0 : Math.log(p<2 ?
                    Math.pow(resp, -alpha)*Math.pow(p-1, alpha)*Math.pow(weight, 1-alpha)/((2-p)*Math.pow(phi, 1-alpha))
                    : Math.pow(resp, -alpha)*Math.pow(p-1, alpha)*Math.pow(weight, 1-alpha)/((p-2)*Math.pow(phi, 1-alpha)));
            manualInfoCols[1] = logZ;
            // generate logWVMax
            manualInfoCols[2] = resp == 0 ? 0 : (p<2 ? jkMax*logZ-Math.log(gamma(1+jkMax))- Math.log(gamma(-alpha*jkMax)) :
                    Math.log(gamma(1+alpha*jkMax))+jkMax*(alpha-1)*Math.log(phi)+alpha*jkMax*Math.log(p-1)-
                            Math.log(gamma(1+jkMax))-jkMax*Math.log(p-2)-alpha*jkMax*Math.log(resp));
            // generate dlogWVMax
            manualInfoCols[3] = resp == 0 ? 0 : Math.log(jkMax)+manualInfoCols[2];
            // generate d2logWVMax
            manualInfoCols[4] = resp == 0 ? 0 : Math.log(2) + manualInfoCols[3];
            // verify that the lower and upper bounds jkL, jkU, djkL, djkU are generated correctly by making sure the
            // indices are where the lower and upper bounds should be
            if (resp != 0)
                assertCorrectBounds(infoColRow[5], infoColRow[6], infoColRow[7], infoColRow[8], manualInfoCols[1], 
                        alpha, p, Math.log(params._tweedie_epsilon), (int) jkMax);
            System.arraycopy(infoColRow, 5, manualInfoCols, 5, 4);
            // cal LL, DLL, D2LL
            double sumWV = resp == 0 ? 0 : calWVSum(p, logZ, alpha, (int) manualInfoCols[5], (int) manualInfoCols[6],
                    resp);  // this one contains 1/y or 1/(pi*y)
            manualInfoCols[9] = sumWV*(p < 2 ? resp : (Math.PI*resp)); // without 1/y or 1/(PI*y)
            double part2 = -Math.pow(mu, 2-p)/(phi*(2-p));
            manualInfoCols[12] = resp==0 ? part2 : resp*Math.pow(mu, 1-p)/((1-p)*phi)+part2+Math.log(sumWV);
            double sumDWV  = resp == 0 ? 0 : calDWV(p, logZ, alpha, (int) manualInfoCols[7], (int) manualInfoCols[8], phi);
            manualInfoCols[10] = sumDWV;
            double dWOverW = resp == 0 ? 0 : sumDWV/manualInfoCols[9];
            double dpart2 = Math.pow(mu, 2-p)/(phi*phi*(2-p));
            manualInfoCols[13] = resp==0?dpart2:dpart2+dWOverW+resp*Math.pow(mu,1-p)/(phi*phi*(p-1));
           double d2part2 = -2*Math.pow(mu, 2-p)/(phi*phi*phi*(2-p));
           double sumD2WV = resp==0 ? 0 : calD2WV(p, logZ, alpha, (int) manualInfoCols[7], (int) manualInfoCols[8], phi);
           manualInfoCols[11] = sumD2WV;
           double d2WVOverdPhi = resp==0 ? 0 : (manualInfoCols[9]*sumD2WV-sumDWV*sumDWV)/(manualInfoCols[9]*manualInfoCols[9]);
           manualInfoCols[14] = resp==0 ? d2part2 : -2*resp*Math.pow(mu, 1-p)/(phi*phi*phi*(p-1))+d2part2+d2WVOverdPhi;
        }
    }

    public double calD2WV(double vPower, double logZ, double alpha, int djkL, int djkU, double phi) {
        if (vPower < 2) {
            double part1 = IntStream.rangeClosed(djkL, djkU).mapToDouble(x ->
                    Math.exp(calD2LogWV(vPower, logZ, alpha, x))).sum() * (1 - alpha) * (1 - alpha) / (phi * phi);
            double part2 = IntStream.rangeClosed(djkL, djkU).mapToDouble(x ->
                    Math.exp(calDLogWV(vPower, logZ, alpha, x))).sum() * (1 - alpha) / (phi * phi);
            return part1 + part2;
        } else {
            double part1 = IntStream.rangeClosed(djkL, djkU).mapToDouble(x ->
                    Math.exp(calD2LogWV(vPower, logZ, alpha, x)) * Math.pow(-1, x) * Math.sin(-x * Math.PI * alpha)).sum() * (1 - alpha) * (1 - alpha) / (phi * phi);
            double part2 = IntStream.rangeClosed(djkL, djkU).mapToDouble(x ->
                    Math.exp(calDLogWV(vPower, logZ, alpha, x)) * Math.pow(-1, x) * Math.sin(-x * Math.PI * alpha)).sum() * (1 - alpha) / (phi * phi);
            return part1 + part2;
        }
    }
    
    public double calDWV(double vPower, double logZ, double alpha,  int DjkL, int DjkU, 
                              double phi) {
        if (vPower < 2)
            return IntStream.rangeClosed(DjkL, DjkU).mapToDouble(x ->
                    Math.exp(calDLogWV(vPower, logZ, alpha, x))).sum() * (alpha - 1) / phi;
        else
            return IntStream.rangeClosed(DjkL, DjkU).mapToDouble(x -> 
                Math.exp(calDLogWV(vPower, logZ, alpha, x))*Math.pow(-1, x)*Math.sin(-x*Math.PI*alpha)).sum() * (alpha - 1) / phi;
    }
    
    public double calWVSum(double vPower, double logZ, double alpha, int jkL, int jkU, double resp) {
        if (vPower < 2) {
            double aYPhi = IntStream.rangeClosed(jkL, jkU).mapToDouble(x -> Math.exp(calLogWV(vPower, logZ, alpha, x))).sum();
            return aYPhi / resp;
        } else {
            double aYPhi = IntStream.rangeClosed(jkL, jkU).mapToDouble(x ->
                    Math.exp(calLogWV(vPower, logZ, alpha, x))*Math.pow(-1, x)*Math.sin(-x*Math.PI*alpha)).sum();
            return aYPhi / (resp * Math.PI);
        }
    }
    
    public void assertCorrectBounds(double jkL, double jkU, double DjkL, double DjkU, double logZ, double alpha, 
                                    double vPower, double logEpsilon, int jkMax) {
        // check jkL, jkU
        double logWVMax = calLogWV(vPower, logZ, alpha, jkMax);
        if (jkL > 1)
            assertCorrectLowerBound(calLogWV(vPower, logZ, alpha, (int) jkL)-logWVMax, 
                    calLogWV(vPower, logZ, alpha, (int) jkL+1)-logWVMax, logEpsilon);
        if (jkU > 1)
            assertCorrectUpperBound(calLogWV(vPower, logZ, alpha, (int) jkU)-logWVMax,
                calLogWV(vPower, logZ, alpha, (int) jkU-1)-logWVMax, logEpsilon);
        
        // check DjkL, DjkU
        double dlogWVMax = calDLogWV(vPower, logZ, alpha, jkMax);
        if (DjkL > 1)
            assertCorrectLowerBound(calDLogWV(vPower, logZ, alpha, (int) DjkL)-dlogWVMax,
                    calDLogWV(vPower, logZ, alpha, (int) DjkL+1)-dlogWVMax, logEpsilon);
        if (DjkU > 1)
            assertCorrectUpperBound(calDLogWV(vPower, logZ, alpha, (int) DjkU)-dlogWVMax,
                    calDLogWV(vPower, logZ, alpha, (int) DjkU-1)-dlogWVMax, logEpsilon);
        // check DjkL, DjkU for d2
        double d2logWVMax = calD2LogWV(vPower, logZ, alpha, jkMax);
        if (DjkL > 1)
            assertCorrectLowerBound(calD2LogWV(vPower, logZ, alpha, (int) DjkL)-d2logWVMax,
                    calD2LogWV(vPower, logZ, alpha, (int) DjkL+1)-d2logWVMax, logEpsilon);
        if (DjkU > 1)
            assertCorrectUpperBound(calD2LogWV(vPower, logZ, alpha, (int) DjkU)-d2logWVMax,
                    calD2LogWV(vPower, logZ, alpha, (int) DjkU-1)-d2logWVMax, logEpsilon);
        
    }
    
    public void assertCorrectLowerBound(double currVal, double nextVal, double logEpsilon) {
        Assert.assertTrue(currVal < logEpsilon && nextVal >= logEpsilon);
    }
    
    public void assertCorrectUpperBound(double currVal, double preVal, double logEpsilon) {
        Assert.assertTrue(currVal < logEpsilon && preVal >= logEpsilon);
    }
    
    public double calLogWV(double vPower, double logZ, double alpha, int index) {
        if (vPower < 2) {
            return index*logZ-logGamma(1+index)-logGamma(-alpha*index);
        } else {
            return index*logZ+logGamma(1+alpha*index)-logGamma(1+index);
        }
    }
    
    public double calDLogWV(double vPower, double logZ, double alpha, int index) {
        if (vPower < 2) {
            return Math.log(index)+index*logZ-logGamma(1+index)-logGamma(-alpha*index);
        } else {
            return Math.log(index)+index*logZ+logGamma(1+alpha*index)-logGamma(1+index);
        }
    }
    
    public double calD2LogWV(double vPower, double logZ, double alpha, int index) {
        if (vPower < 2) {
            return Math.log(index)+index*logZ-logGamma(1+index)-logGamma(-alpha*index)+Math.log(2);
        } else {
            return Math.log(index)+index*logZ+logGamma(1+alpha*index)-logGamma(1+index)+Math.log(2);
        }
    }
    
    public void assertConstColGeneration(Frame train, GLMModel.GLMParameters params) {
        params._train = train._key;
        params._compute_p_values = true;
        params._dispersion_factor_method = pearson;
        params._family = tweedie;
        params._lambda = new double[]{0.0};
        GLMModel glmML = new GLM(params).trainModel().get();
        Scope.track_generic(glmML);

        TweedieMLDispersionOnly mlDisp = new TweedieMLDispersionOnly(train, params, glmML, glmML._output._job);
        compareConstFrame(params, mlDisp);
        mlDisp.cleanUp();
    }
    
    public void compareConstFrame(GLMModel.GLMParameters params, TweedieMLDispersionOnly mlDisp) {
        Frame infoFrame = mlDisp._infoFrame;
        Scope.track(infoFrame);
        String[] constColNames = mlDisp._constFrameNames;
        int numRows = (int) infoFrame.numRows();
        int numCols = constColNames.length;
        double[] manualConstRow = new double[numCols];
        double[] constRow = new double[numCols];

        int offset = mlDisp._weightPresent ? 3 : 2;
        for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
            manuallyGenerateConst(manualConstRow, mlDisp, params, rowIndex);
            extractFrame2Array(constRow, infoFrame, offset, rowIndex);
            Assert.assertTrue(TestUtil.equalTwoArrays(manualConstRow, constRow, 1e-6));
        }
    }
    
    public void extractFrame2Array(double[] constRow, Frame infoFrame, int colOffset, int rowInd) {
        int numCols = constRow.length;
        for (int colInd = 0; colInd < numCols; colInd++)
            constRow[colInd] = infoFrame.vec(colInd+colOffset).at(rowInd);
    }
    
    public void manuallyGenerateConst(double[] manualConstRow, TweedieMLDispersionOnly mlDisp,
                                      GLMModel.GLMParameters params, int rowInd) {
        Frame respMu = mlDisp._infoFrame;
        Scope.track(respMu);
        double resp = respMu.vec(0).at(rowInd);
        double mu = respMu.vec(1).at(rowInd);
        double weight = mlDisp._weightPresent ? respMu.vec(2).at(rowInd) : 1;
        double p = params._tweedie_variance_power;
        double alpha = (2.0-p)/(1.0-p);
        if (!Double.isNaN(resp)) {
            // calculate jMaxConst
            if (resp != 0)
                manualConstRow[0] = p<2 ? weight*Math.pow(resp, 2-p)/(2-p) : weight*Math.pow(resp, 2-p)/(p-2);
            else
                manualConstRow[0] = Double.NaN;
            // calculate zConst
            manualConstRow[1] = resp == 0 ? 0 : Math.pow(weight, 1-alpha)*Math.pow(resp, -alpha)* 
                    Math.pow(p-1, alpha)/(2-p);
            // calculate part2Const
            if (resp==0.0) {
                manualConstRow[2] = -weight*Math.pow(mu, 2-p)/(2-p);
            } else {
                manualConstRow[2] = weight*resp*Math.pow(mu, 1-p)/(1-p)-weight*Math.pow(mu, 2-p)/(2-p);
            }
            // calculate oneOverY
            manualConstRow[3] = resp==0 ? Double.NaN : Math.log(1.0/resp);
            // calculate oneOverPiY
            manualConstRow[4] = resp==0 ? Double.NaN : Math.log(1.0/(resp*Math.PI));
            // calculate firstOrderDerivConst
            manualConstRow[5] = Math.pow(weight, 2)*resp*Math.pow(mu, 1-p)/(p-1)+Math.pow(weight, 2)*Math.pow(mu, 2-p)/(2-p);
            // calculate secondOrderDerivConst
            manualConstRow[6] = -2*Math.pow(weight, 3)*resp*Math.pow(mu, 1-p)/(p-1)-2*Math.pow(weight, 3)*Math.pow(mu, 2-p)/(2-p);
        } else {
            Arrays.fill(manualConstRow, Double.NaN);
        }
    }
}
