package hex.modelselection;

import Jama.Matrix;
import hex.DataInfo;
import org.junit.Test;
import org.junit.runner.RunWith;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;
import water.runner.CloudSize;
import water.runner.H2ORunner;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static hex.glm.GLMModel.GLMParameters.Family.gaussian;
import static hex.modelselection.ModelSelectionModel.ModelSelectionParameters.Mode.maxr;
import static hex.modelselection.ModelSelectionModel.ModelSelectionParameters.Mode.maxrsweep;
import static hex.modelselection.ModelSelectionUtils.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

@RunWith(H2ORunner.class)
@CloudSize(1)
public class ModelSelectionMaxRSweepTests extends TestUtil {
    
    @Test
    public void testAllSweepingVectors() {
        final double[][] cpm = new double[][]{{10.000000, -3.925113,  1.192877,  3.657043},
                {-3.925113,  9.961056,  4.638419, -2.373317},
                {1.192877,  4.638419,  9.649718, 3.444272},
                {3.657043, -2.373317,  3.444272, 3.636824}};
        final double[][] correctSweepVectors0s1s2 = new double[][]{{0.1000000, -0.3925113,  0.1192877,  0.1000000,  
                0.3657043, -0.1000000, -0.3925113,  0.11928773, 0.1000000,  0.3657043}, {-0.0466143,  0.1187591,  
                0.6064598,  0.1187591,  -0.1113825666211721, 0.0466143, -0.1187591,  0.6064598,  0.1187591, 
                -0.1113825666211721}, {0.05574177,  0.09460483,  0.15599522,  0.15599522, 0.5579671497838223, 
                -0.05574177, -0.09460483,  -0.15599522,  0.15599522, 0.5579671497838223}};
        // test sweeping vectors for sweep 0, 1, 2
        assertCorrectAllSweepVectors(clone2DArray(cpm), new int[]{0,1,2}, correctSweepVectors0s1s2);
        // test sweeping vectors for sweep 2, 1, 0
        final double[][] correctSweepVectors2s1s0 = new double[][]{{0.1236178,  0.4806792,  0.1036300,  0.1036300,
                0.3569298,  0.1236178,  0.4806792, -0.1036300,  0.1036300,  0.3569298},{-0.58184375,  0.12934160,
                0.06217182,  0.12934160, -0.52110530, -0.58184375, -0.12934160, -0.06217182,  0.12934160, -0.52110530},
                {0.13821485, -0.08041945,  0.05574177,  0.13821485,  0.12260698, -0.13821485,  0.08041945, -0.05574177,
                        0.13821485,  0.12260698}};
        assertCorrectAllSweepVectors(clone2DArray(cpm), new int[]{2,1,0}, correctSweepVectors2s1s0);
        // test sweeping vectors for 2, 0, 1
        
        final double[][] correctSweepVectors2s0s1 = new double[][]{{-0.3940459,  0.1003910,  0.4656553,  0.1003910,
                -0.2382596, -0.3940459, -0.1003910,  0.4656553,  0.1003910, -0.2382596},{0.1182966, -0.0466143,
                0.3573300,  0.1182966,  0.3219854, -0.1182966,  0.0466143,  0.3573300,  0.1182966,  0.3219854},
                {0.05574177,  0.09460483,  0.15599522,  0.15599522,  0.55796715, -0.05574177, -0.09460483, -0.15599522,
                        0.15599522,  0.55796715}};
        assertCorrectAllSweepVectors(clone2DArray(cpm), new int[]{1,0,2}, correctSweepVectors2s0s1);
    }

    public static void assertCorrectAllSweepVectors(double[][] cpm, int[] sweepIndices, double[][] rSweepVector) {
        int numSweepVec = 2*(cpm.length+1);
        SweepVector[][] sweepVecs =  sweepCPM(cpm, sweepIndices, true);
        int numSweep = sweepIndices.length;
        for (int sweepInd=0; sweepInd < numSweep; sweepInd++) {
            for (int index = 0; index < numSweepVec; index++) {
                assertTrue(Math.abs(rSweepVector[sweepInd][index] - sweepVecs[sweepInd][index]._value) < 1e-6);
            }
        }
    }
    // test to make sure sweeping vectors are generated correctly for one sweep at a time.
    @Test 
    public void testSweepingVectorGeneration4OneSweep() {
        final double[][] cpm = new double[][]{{10.000000, -3.925113,  1.192877,  3.657043},
                {-3.925113,  9.961056,  4.638419, -2.373317},
                {1.192877,  4.638419,  9.649718, 3.444272},
                {3.657043, -2.373317,  3.444272, 3.636824}};
        // after sweep 0
        final double[] sweep0Vector = new double[]{0.1000000, -0.3925113,  0.1192877,  0.1000000,  0.3657043, -0.1000000,
                -0.3925113,  0.11928773, 0.1000000,  0.3657043};
        assertCorrectOneSweepVectors(cpm, 0, sweep0Vector);
        // after sweep 1
        final double[] sweep1Vector = new double[]{-0.0466143,  0.1187591,  0.6064598,  0.1187591,  -0.1113825666211721,
                0.0466143, -0.1187591,  0.6064598,  0.1187591, -0.1113825666211721};
        assertCorrectOneSweepVectors(cpm, 1, sweep1Vector);
        // after sweep 2
        final double[] sweep2Vector = new double[]{0.05574177,  0.09460483,  0.15599522,  0.15599522, 
                0.5579671497838223, -0.05574177, -0.09460483,  -0.15599522,  0.15599522, 0.5579671497838223};
        assertCorrectOneSweepVectors(cpm, 2, sweep2Vector);
    }
    
    @Test
    public void testApplyOneSweepVectorOneNewRowCol() {
        final double[][] cpm = new double[][]{{10.000000, -3.925113,  1.192877, -2.675484,  3.657043},
                {-3.925113,  9.961056,  4.638419,  1.719902, -2.373317},
                {1.192877,  4.638419,  9.649718, -1.855625,  3.444272},
                {-2.675484,  1.719902, -1.855625,  6.197150, -3.117239},
                {3.657043, -2.373317,  3.444272, -3.117239,  3.636824}};
        int[][] predInd2CPMMap = new int[][]{{0},{1},{2},{3}};
        // without intercepts
        // after sweep 0 and adding only one col/row
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{0}, new int[]{0,1,2}, 3, new int[]{0,1,2,3},
                predInd2CPMMap, false);
        // after sweep 1 and adding only one col/row
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{1}, new int[]{0,1,2}, 3, new int[]{0,1,2,3},
                predInd2CPMMap, false);
        // after sweep 2 and adding only one col/row
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{2}, new int[]{0,1,2}, 3, new int[]{0,1,2,3},
                predInd2CPMMap, false);

        // with intercepts
        // after sweep 0 and adding only one col/row
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{0}, new int[]{0,1}, 3, new int[]{0,1,2},
                predInd2CPMMap, true);
        // after sweep 1 and adding only one col/row
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{1}, new int[]{0,1}, 3, new int[]{0,1,2},
                predInd2CPMMap, true);
        // after sweep 2 and adding only one col/row
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{2}, new int[]{0,1}, 3, new int[]{0,1,2},
                predInd2CPMMap, true);
    }
    
    @Test
    public void testApplyOneSweepVectorMultipleNewRowsCols() {
        final double[][] cpm = new double[][]{{10.000000, -3.925113,  1.192877, -2.675484,  3.657043},
                {-3.925113,  9.961056,  4.638419,  1.719902, -2.373317},
                {1.192877,  4.638419,  9.649718, -1.855625,  3.444272},
                {-2.675484,  1.719902, -1.855625,  6.197150, -3.117239},
                {3.657043, -2.373317,  3.444272, -3.117239,  3.636824}};
        // without intercepts
        // after sweep 1 and adding two row/cols
        int[][] predInd2CPMMap = new int[][]{{0},{1},{2, 3}};
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{1}, new int[]{1,0}, 2, new int[]{1,0,2},
                predInd2CPMMap, false);
        // after sweep 0 and adding three row/cols
        predInd2CPMMap = new int[][]{{0},{1, 2, 3}};
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{0}, new int[]{0}, 1, new int[]{0,1},
                predInd2CPMMap, false);

        // with intercepts
        // after sweep 1 and adding two row/cols
        predInd2CPMMap = new int[][]{{1,2},{0}};
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{1}, new int[]{1}, 0, new int[]{1,0},
                predInd2CPMMap, true);
        // after sweep 0 and adding three row/cols
        predInd2CPMMap = new int[][]{{0},{1, 2}};
        assertCorrectOneSweepVectorApplication(clone2DArray(cpm), new int[]{0}, new int[]{0}, 1, new int[]{0,1},
                predInd2CPMMap, true);
    }

    @Test
    public void testApplyMultipleSweepVectorsMultipleNewRowsCols() {
        final double[][] allCPM = new double[][]{{10.0000000,  1.2775780, -2.3233446,  5.5573236, -2.6207931,
                -0.8321198, -2.275524}, {1.2775780,  6.9109954, -0.6907638,  1.6104894, -0.2288084,  1.0167696,
                -4.049839}, {-2.3233446, -0.6907638,  2.9167538, -0.9018913, -0.8793179, -1.6471318,  2.448738},
                {5.5573236,  1.6104894, -0.9018913, 11.4569287, -1.9699385, -2.2898801, -2.018988}, {-2.6207931,
                -0.2288084, -0.8793179, -1.9699385,  6.9525541, -0.2876347,  2.896199}, {-0.8321198,  1.0167696,
                -1.6471318, -2.2898801, -0.2876347, 11.3756280, -8.946720}, {-2.2755243, -4.0498392,  2.4487380,
                -2.0189880,  2.8961986, -8.9467197, 10.464016}};
        // without intercept
        int[][] predInd2CPMMap = new int[][]{{0},{1,2},{3,4,5}};
        assertCorrectMultipleSVectors(clone2DArray(allCPM), new int[]{0,1,2,3,4}, new int[]{1,2}, 0, new int[]{1,2,0},
                predInd2CPMMap, false);
        assertCorrectMultipleSVectors(clone2DArray(allCPM), new int[]{0,1,2}, new int[]{0,1}, 2, new int[]{0,1,2},
                predInd2CPMMap, false);
        assertCorrectMultipleSVectors(clone2DArray(allCPM), new int[]{0,1,2,3}, new int[]{0,2}, 1, new int[]{0,2,1},
                predInd2CPMMap, false);
       
        // with intercept
        predInd2CPMMap = new int[][]{{0,1,2},{3,4}};
        assertCorrectMultipleSVectors(clone2DArray(allCPM), new int[]{0,1,2}, new int[]{0}, 1, new int[]{0,1},
                predInd2CPMMap, true);
        assertCorrectMultipleSVectors(clone2DArray(allCPM), new int[]{0,1}, new int[]{1}, 0, new int[]{1,0},
                predInd2CPMMap, true);
        assertCorrectMultipleSVectors(clone2DArray(allCPM), new int[]{0,1,2}, new int[]{0}, 1, new int[]{0,1},
                predInd2CPMMap, true);
    }
    
    public static void assertCorrectMultipleSVectors(double[][] allCPM, int[] sweepIndices, int[] predIndices, 
                                                     int newPredInd, int[] allPreds, int[][] pred2CPMIndices, 
                                                     boolean hasIntercept) {
        // generate correct CPM after all sweeps are performed
        double[][] correctCPM = extractPredSubsetsCPM(allCPM, allPreds, pred2CPMIndices, hasIntercept);
        sweepCPM(correctCPM, sweepIndices, false);
        
        int newPredCPMLength = pred2CPMIndices[newPredInd].length;
        double[][] smallCPM = extractPredSubsetsCPM(allCPM, predIndices, pred2CPMIndices, hasIntercept);
        SweepVector[][] sweepVectors = sweepCPM(smallCPM, sweepIndices, true);
        double[][] rightSizeCPM = addNewPred2CPM(allCPM, smallCPM, allPreds, pred2CPMIndices, hasIntercept);
        if (newPredCPMLength == 1) {
            applySweep2NewPred(sweepVectors, rightSizeCPM, newPredCPMLength);
        } else {
            SweepVector[][] newSweepVec = mapBasicVector2Multiple(sweepVectors, newPredCPMLength);
            applySweep2NewPred(newSweepVec, rightSizeCPM, newPredCPMLength);
        }
        assert2DArraysEqual(correctCPM, rightSizeCPM, 1e-6);
    }
    
    public static void assertCorrectOneSweepVectorApplication(double[][] allCPM, int[] sweepIndices, int[] predIndices,
                                                              int newPredInd, int[] allPreds, int[][] pred2CPMIndices,
                                                              boolean hasIntercept) {
        // generate correct CPM after all sweeps are performed
        double[][] correctCPM = extractPredSubsetsCPM(allCPM, allPreds, pred2CPMIndices, hasIntercept);
        sweepCPM(correctCPM, sweepIndices, false);
        
        // generate CPM after sweeping and correct application of sweep vector
        int newPredCPMLength = pred2CPMIndices[newPredInd].length;
        double[][] smallCPM = extractPredSubsetsCPM(allCPM, predIndices, pred2CPMIndices, hasIntercept);
        SweepVector[][] sweepVectors = sweepCPM(smallCPM, sweepIndices, true);
        double[][] rightSizeCPM = addNewPred2CPM(allCPM, smallCPM, allPreds, pred2CPMIndices, hasIntercept);
        if (pred2CPMIndices[newPredInd].length > 1) {
            SweepVector[][] newSweepVec = mapBasicVector2Multiple(sweepVectors, newPredCPMLength);
            oneSweepWSweepVector(newSweepVec[0], rightSizeCPM, sweepIndices[0], newPredCPMLength);
        } else {
            oneSweepWSweepVector(sweepVectors[0], rightSizeCPM, sweepIndices[0], newPredCPMLength);
        }
        // compare CPM generated manually and by program
        assert2DArraysEqual(correctCPM, rightSizeCPM, 1e-6);
    }
    
    public static void assertCorrectOneSweepVectors(double[][] cpm, int sweepIndex, double[] rSweepVector) {
        int numSweepVec = 2*(cpm.length+1);
        SweepVector[] sweepVec = new SweepVector[numSweepVec];
        performOneSweep(cpm, sweepVec, sweepIndex, true);

        for (int index=0; index<numSweepVec; index++) {
            assertTrue(Math.abs(rSweepVector[index]-sweepVec[index]._value)<1e-6);
        }
    }
    
    @Test
    public void testPerformAllSweepingNoSweepVector() {
        final double[][] cpm = new double[][]{{10.000000, -3.925113,  1.192877, -2.675484,  3.657043},
                {-3.925113,  9.961056,  4.638419,  1.719902, -2.373317},
                {1.192877,  4.638419,  9.649718, -1.855625,  3.444272},
                {-2.675484,  1.719902, -1.855625,  6.197150, -3.117239},
                {3.657043, -2.373317,  3.444272, -3.117239,  3.636824}};
        final double[][] cpmAfterSweep0N1 = new double[][]{{0.1182966,  0.04661430,  0.3573300, -0.23632874,  0.3219854},
                {0.0466143,  0.11875914,  0.6064598,  0.07953825, -0.1113826},
                {-0.3573300, -0.60645976,  6.4104528, -1.94264564 , 3.5768221},
                {0.2363287, -0.07953825, -1.9426456,  5.42805824, -2.0642052},
                {-0.3219854,  0.11138257,  3.5768221, -2.06420516, 2.1949635}};
        assertCorrectAllSweeping(clone2DArray(cpm), new int[]{0, 1}, cpmAfterSweep0N1);
        assertCorrectAllSweeping(clone2DArray(cpm), new int[]{1, 0}, cpmAfterSweep0N1);
        final double[][] cpmAfterSweep0N1N2 = new double[][]{{0.13821485,  0.08041945, -0.05574177, -0.1280422,  0.1226070},
                {0.08041945,  0.17613316, -0.09460483,  0.2633219, -0.4497672},
                {-0.05574177, -0.09460483,  0.15599522, -0.3030434,  0.5579671},
                {0.12804222, -0.26332191,  0.30304344,  4.8393522, -0.9802727},
                {-0.12260698,  0.44976719, -0.55796715, -0.9802727,  0.1992143}};
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,1,2}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,2,1}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{2,0,1}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{2,1,0}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1,2,0}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1,0,2}, cpmAfterSweep0N1N2);
        final double[][] cpmAfterSweep0N1N2N3 = new double[][]{{0.14160266,  0.07345233, -0.04772369,  0.02645855,  0.0966703864},
                {0.07345233,  0.19046119, -0.11109422, -0.05441263, -0.3964279716},
                {-0.04772369, -0.11109422,  0.17497200,  0.06262066,  0.4965818245},
                {0.02645855, -0.05441263,  0.06262066,  0.20663923, -0.2025627940},
                {-0.09667039,  0.39642797, -0.49658182,  0.20256279,  0.0006474807}};
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,1,2,3}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,2,1,3}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{3,2,1,0}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{3,1,2,0}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1,0,2,3}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1,3,0,2}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{2,3,1,0}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{2,1,0,3}, cpmAfterSweep0N1N2N3);
    }
    
    public static void assertCorrectAllSweeping(double[][] cpm, int[] sweepIndice, double[][] correctCPM) {
        sweepCPM(cpm, sweepIndice, false);
        assert2DArraysEqual(correctCPM, cpm, 1e-6);
    }
    
    // test perform one sweep without considering sweeping vectors.  We test with sweeping one index at a time, 
    // multiple indice.  Sweeping action is commutative and I will be testing that with multiple indice with indice
    // permuted.
    @Test
    public void testPerformOneSweepNoSweepVector() {
        final double[][] cpm = new double[][]{{10.000000, -3.925113,  1.192877, -2.675484,  3.657043}, 
                {-3.925113,  9.961056,  4.638419,  1.719902, -2.373317},
                {1.192877,  4.638419,  9.649718, -1.855625,  3.444272},
                {-2.675484,  1.719902, -1.855625,  6.197150, -3.117239}, 
                {3.657043, -2.373317,  3.444272, -3.117239,  3.636824}};
        final double[][] cpmAfterSweep0 = new double[][]{{0.1000000, -0.3925113,  0.1192877, -0.2675484,  0.3657043},
                {0.3925113,  8.4204048,  5.1066367,  0.6697443, -0.9378863},
                {-0.1192877,  5.1066367,  9.5074224, -1.5364727,  3.0080318},
                {0.2675484,  0.6697443, -1.5364727,  5.4813285, -2.1388030},
                {-0.3657043, -0.9378863,  3.0080318, -2.1388030,  2.2994276}};
        double[][] cpmClone = clone2DArray(cpm);
        assertCorrectSweeping(cpmClone, new int[]{0}, cpmAfterSweep0);
        final double[][] cpmAfterSweep0N1 = new double[][]{{0.1182966,  0.04661430,  0.3573300, -0.23632874,  0.3219854},
                {0.0466143,  0.11875914,  0.6064598,  0.07953825, -0.1113826},
                {-0.3573300, -0.60645976,  6.4104528, -1.94264564 , 3.5768221},
                {0.2363287, -0.07953825, -1.9426456,  5.42805824, -2.0642052},
                {-0.3219854,  0.11138257,  3.5768221, -2.06420516, 2.1949635}};
        assertCorrectSweeping(cpmClone, new int[]{1}, cpmAfterSweep0N1);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,1}, cpmAfterSweep0N1);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1, 0}, cpmAfterSweep0N1);
        final double[][] cpmAfterSweep0N1N2 = new double[][]{{0.13821485,  0.08041945, -0.05574177, -0.1280422,  0.1226070},
                {0.08041945,  0.17613316, -0.09460483,  0.2633219, -0.4497672},
                {-0.05574177, -0.09460483,  0.15599522, -0.3030434,  0.5579671},
                {0.12804222, -0.26332191,  0.30304344,  4.8393522, -0.9802727},
                {-0.12260698,  0.44976719, -0.55796715, -0.9802727,  0.1992143}};
        assertCorrectSweeping(cpmClone, new int[]{2}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,1,2}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,2,1}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{2,0,1}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{2,1,0}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1,2,0}, cpmAfterSweep0N1N2);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1,0,2}, cpmAfterSweep0N1N2);
        final double[][] cpmAfterSweep0N1N2N3 = new double[][]{{0.14160266,  0.07345233, -0.04772369,  0.02645855,  0.0966703864},
                {0.07345233,  0.19046119, -0.11109422, -0.05441263, -0.3964279716},
                {-0.04772369, -0.11109422,  0.17497200,  0.06262066,  0.4965818245},
                {0.02645855, -0.05441263,  0.06262066,  0.20663923, -0.2025627940},
                {-0.09667039,  0.39642797, -0.49658182,  0.20256279,  0.0006474807}};
        assertCorrectSweeping(cpmClone, new int[]{3}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,1,2,3}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{0,2,1,3}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{3,2,1,0}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{3,1,2,0}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1,0,2,3}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{1,3,0,2}, cpmAfterSweep0N1N2N3);
        assertCorrectSweeping(clone2DArray(cpm), new int[]{2,3,1,0}, cpmAfterSweep0N1N2N3);        
        assertCorrectSweeping(clone2DArray(cpm), new int[]{2,1,0,3}, cpmAfterSweep0N1N2N3);
    }
    
    public static double[][] clone2DArray(double[][] cpm) {
        int cpmDim = cpm.length;
        double[][] cpmClone = new double[cpmDim][cpmDim];
        for (int index=0; index<cpmDim; index++)
            System.arraycopy(cpm[index], 0, cpmClone[index], 0, cpmDim);
        return cpmClone;
    }
    
    public static void assert2DArraysEqual(double[][] array1, double[][] array2, double tot) {
        int dimArray = array1.length;
        for (int index=0; index<dimArray; index++)
            assertArrayEquals("expected and extracted cross-product swept matrice are not equal",
                    array1[index], array2[index], tot);
    }
    
    public static void assertCorrectSweeping(double[][] cpm, int[] sweepIndice, double[][] correctCPM) {
        int numSweep = sweepIndice.length;
        for (int index=0; index<numSweep; index++)
            performOneSweep(cpm, null, sweepIndice[index], false);

        assert2DArraysEqual(correctCPM, cpm, 1e-6);
    }
    
    @Test
    public void testExtractSubsetCPM() {
        final double[] vecValues = new double[]{1, 0.1, 0.2, 0.3, 0.4};
        final double[] vecValues2 = new double[]{0.1, 0.2, 0.3, 0.4, 1};
        final double[] resp = new double[]{2};
        final double[][] allCPM = generateCPM(vecValues2, resp);
        final double[][] allCPMInterceptFirst = generateCPM(vecValues, resp);
        
        // tests with intercept
        assertCorrectCPMExtraction(allCPM, new int[][]{{0},{},{},{}}, allCPMInterceptFirst, new int[]{0}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{0, 1},{},{},{}}, allCPMInterceptFirst, new int[]{0}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{0, 1, 2},{},{},{}}, allCPMInterceptFirst, new int[]{0}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{0, 1, 2, 3},{},{},{}}, allCPMInterceptFirst, new int[]{0}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{1}, {}, {}}, allCPMInterceptFirst, new int[]{1}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{1, 2}, {}, {}}, allCPMInterceptFirst, new int[]{1}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{1, 2, 3}, {}, {}}, allCPMInterceptFirst, new int[]{1}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{}, {2}, {}}, allCPMInterceptFirst, new int[]{2}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{}, {2, 3}, {}}, allCPMInterceptFirst, new int[]{2}, true);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{}, {}, {3}}, allCPMInterceptFirst, new int[]{3}, true);   
        
        // tests with intercept
        assertCorrectCPMExtraction(allCPM, new int[][]{{0}, {}, {}, {}, {}}, allCPM, new int[]{0}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{0, 1}, {}, {}, {}, {}}, allCPM, new int[]{0}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{0, 1, 2}, {}, {}, {}, {}}, allCPM, new int[]{0}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{0, 1, 2, 3}, {}, {}, {}, {}}, allCPM, new int[]{0}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{0, 1, 2, 3, 4}, {}, {}, {}, {}}, allCPM, new int[]{0}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{1}, {}, {}, {}}, allCPM, new int[]{1}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{1, 2}, {}, {}, {}}, allCPM, new int[]{1}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{1, 2, 3}, {}, {}, {}}, allCPM, new int[]{1}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{1, 2, 3, 4}, {}, {}, {}}, allCPM, new int[]{1}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{}, {2}, {}, {}}, allCPM, new int[]{2}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{}, {2, 3}, {}, {}}, allCPM, new int[]{2}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{}, {2,3,4}, {}, {}}, allCPM, new int[]{2}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{}, {}, {}, {3}, {}}, allCPM, new int[]{3}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{},{}, {}, {3,4}, {}}, allCPM, new int[]{3}, false);
        assertCorrectCPMExtraction(allCPM, new int[][]{{}, {}, {}, {}, {4}, {}}, allCPM, new int[]{4}, false);
    }
    
    public static double[][] generateCPM(double[] vec, double[] resp) {
        int predLen = vec.length;
        int cpmSize = predLen+1;
        Matrix vecValue = new Matrix(vec, vec.length);
        Matrix respV = new Matrix(resp, 1);
        double[][] xtx = vecValue.times(vecValue.transpose()).getArray();
        double[][] xty = vecValue.times(new Matrix(resp, 1)).getArray();
        double[][] ytx = respV.times(vecValue.transpose()).getArray();
        double yty = respV.times(respV).getArray()[0][0];
        double[][] genCPM = new double[cpmSize][cpmSize];
        int xtxLen = xtx.length;
        for (int index=0; index<xtxLen; index++) {
            System.arraycopy(xtx[index], 0, genCPM[index], 0, xtxLen);
            genCPM[index][xtxLen] = xty[index][0];
        }
        System.arraycopy(ytx[0], 0, genCPM[xtxLen], 0, xtxLen);
        genCPM[xtxLen][xtxLen] = yty;
        return genCPM;
    }
    
    public static void assertCorrectCPMExtraction(double[][] allCPM, int[][] pred2CMPIndices, 
                                                  double[][] CPMInterceptFirst, int[] predIndices, boolean hasIntercept) {
        double[][] extractedCpm = extractPredSubsetsCPM(allCPM, predIndices, pred2CMPIndices,
                hasIntercept);
        List<Integer> predIndicesList = Arrays.stream(pred2CMPIndices[predIndices[0]]).boxed().collect(Collectors.toList());
        if (hasIntercept) {
            predIndicesList = Arrays.stream(pred2CMPIndices[predIndices[0]]).map(x -> x + 1).boxed().collect(Collectors.toList());
            predIndicesList.add(0,0);
        }
        int allCPMSize = allCPM.length;
        predIndicesList.add(allCPMSize-1);
        int cpmDim = predIndicesList.size();
        double[][] correctCPM = new double[cpmDim][cpmDim];
        // generate correct matrix
        for (int rIndex=0; rIndex<cpmDim; rIndex++) {
            for (int cIndex=rIndex; cIndex<cpmDim; cIndex++) {
                correctCPM[rIndex][cIndex] = CPMInterceptFirst[predIndicesList.get(rIndex)][predIndicesList.get(cIndex)];
                correctCPM[cIndex][rIndex] = CPMInterceptFirst[predIndicesList.get(cIndex)][predIndicesList.get(rIndex)];
            }
        }
        assert2DArraysEqual(correctCPM, extractedCpm, 1e-6);
    }

    /**
     * createCrossProductMatrix used GLMTask.GLMIterationTask which has been tested before.  We will test and make sure
     * the final CPM is of the right dimension
     */
    @Test
    public void testCreateCrossProductMatrix() {
        Scope.enter();
        try {
            Frame origF = Scope.track(parseTestFile("smalldata/glm_test/gaussian_20cols_10000Rows.csv"));
            int[] eCol = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            Arrays.stream(eCol).forEach(x -> origF.replace(x, origF.vec(x).toCategoricalVec()).remove());
            DKV.put(origF);
            Scope.track(origF);
            ModelSelectionModel.ModelSelectionParameters parms = new ModelSelectionModel.ModelSelectionParameters();
            parms._response_column = "C21";
            parms._family = gaussian;
            parms._max_predictor_number = 1;
            parms._train = origF._key;
            parms._mode = maxrsweep;      
            ModelSelection model = new ModelSelection(parms);
            ModelSelectionModel model1 = model.trainModel().get();
            Scope.track_generic(model1);
        } finally {
            Scope.exit();
        }
    }

/*    *//**
     * This test will test both applySweep2LastPred, oneSweepWSweepVector and mapBasicVector2Multiple
     *//*
    @Test
    public void testApplySweep2LastPred() {
        final double[][] allCPM = new double[][]{{10.0000000,  1.2775780, -2.3233446,  5.5573236, -2.6207931,
                -0.8321198, -2.275524}, {1.2775780,  6.9109954, -0.6907638,  1.6104894, -0.2288084,  1.0167696,
                -4.049839}, {-2.3233446, -0.6907638,  2.9167538, -0.9018913, -0.8793179, -1.6471318,  2.448738},
                {5.5573236,  1.6104894, -0.9018913, 11.4569287, -1.9699385, -2.2898801, -2.018988}, {-2.6207931,
                -0.2288084, -0.8793179, -1.9699385,  6.9525541, -0.2876347,  2.896199}, {-0.8321198,  1.0167696,
                -1.6471318, -2.2898801, -0.2876347, 11.3756280, -8.946720}, {-2.2755243, -4.0498392,  2.4487380,
                -2.0189880,  2.8961986, -8.9467197, 10.464016}};
        int[][] predInd2CPMIndices = new int[][]{{0, 1, 2}, {3, 4}, {5}};
        // without intercept
        // original pred 0, 1, add 2
        assertCorrectSweepVectorApplication(allCPM, predInd2CPMIndices, new int[]{0,1,2,3,4}, new int[]{0,1},
                2, false);
        // original pred 1, 2, add 0
        assertCorrectSweepVectorApplication(allCPM, predInd2CPMIndices, new int[]{0,1,2}, new int[]{1,2}, 0,
                false);
        // original pred 2, add 0
        assertCorrectSweepVectorApplication(allCPM, predInd2CPMIndices, new int[]{0}, new int[]{2},
                0, false);
        
        // with intercept
        predInd2CPMIndices = new int[][]{{0,1,2}, {3,4}};
        // original pred 0, add 1
        // original pred 1, add 0

    }

    public static void assertCorrectSweepVectorApplication(double[][] allCPM, int[][] pred2CPMMap, int[] sweepIndices, 
                                                           int[] predIndices, int newPredInd, boolean hasIntercept) {
        // manually generated correct swept cpm
        int[] allPred = new int[predIndices.length+1];
        System.arraycopy(predIndices, 0, allPred, 0, predIndices.length);
        allPred[predIndices.length] = newPredInd;
        double[][] correctSweptCPM = extractPredSubsetsCPM(allCPM, allPred, pred2CPMMap, hasIntercept);
        sweepCPM(correctSweptCPM, sweepIndices, false);
        
        // swept cpm generated from sweep vectors
        double[][] prevCPM = extractPredSubsetsCPM(allCPM, predIndices, pred2CPMMap, hasIntercept);
        SweepVector[][] sweepVectors = sweepCPM(prevCPM, sweepIndices, true);
        double[][] subsetCPM = addNewPred2CPM(allCPM, prevCPM, allPred, pred2CPMMap, hasIntercept);
        int newPredCPMLength = pred2CPMMap[newPredInd].length;
        if (newPredCPMLength == 1) {
            applySweep2LastPred(sweepVectors, subsetCPM, newPredCPMLength);
        } else {
            SweepVector[][] newSweepVec = mapBasicVector2Multiple(sweepVectors, newPredCPMLength);
            applySweep2LastPred(newSweepVec, subsetCPM, newPredCPMLength);
        }
        int dim = correctSweptCPM.length;
        for (int index=0; index<dim; index++)
            assertArrayEquals("expected and extracted predInd2CPMIndices arrays are not equal",
                    correctSweptCPM[index], subsetCPM[index], 1e-6);
    }*/
    
    @Test
    public void testAddNewPred2CPM() {
        final double[][] allCPM = new double[][]{{10.0000000,  1.2775780, -2.3233446,  5.5573236, -2.6207931,
                -0.8321198, -2.275524}, {1.2775780,  6.9109954, -0.6907638,  1.6104894, -0.2288084,  1.0167696,
                -4.049839}, {-2.3233446, -0.6907638,  2.9167538, -0.9018913, -0.8793179, -1.6471318,  2.448738},
                {5.5573236,  1.6104894, -0.9018913, 11.4569287, -1.9699385, -2.2898801, -2.018988}, {-2.6207931,
                -0.2288084, -0.8793179, -1.9699385,  6.9525541, -0.2876347,  2.896199}, {-0.8321198,  1.0167696,
                -1.6471318, -2.2898801, -0.2876347, 11.3756280, -8.946720}, {-2.2755243, -4.0498392,  2.4487380,
                -2.0189880,  2.8961986, -8.9467197, 10.464016}};
        int[][] predInd2CPMIndices = new int[][]{{0, 1, 2}, {3, 4}, {5}};
        boolean hasIntercept = false;
        // pred 0 is chosen and we want to add predictor 1
        assertCorrectAddNewPred2CPM(allCPM, new int[]{0,1,2}, predInd2CPMIndices, new int[]{0}, 1, hasIntercept);
        // pred 2, 1 are added, want to add pred 0
        assertCorrectAddNewPred2CPM(allCPM, new int[]{0,1,2}, predInd2CPMIndices, new int[]{2,1}, 0, hasIntercept);
        // pred 0, 2 are added want to add pred 1
        assertCorrectAddNewPred2CPM(allCPM, new int[]{0,1,2,3}, predInd2CPMIndices, new int[]{0,2}, 1, hasIntercept);
        // with intercept
        predInd2CPMIndices = new int[][]{{0,1},{2},{3},{4}};
        hasIntercept = true;
        // pred 3,2 are chosen, add 0
        assertCorrectAddNewPred2CPM(allCPM, new int[]{0,1,2}, predInd2CPMIndices, new int[]{3,2}, 0, hasIntercept);
        // pred 2,3,1 are chosen, add 0
        assertCorrectAddNewPred2CPM(allCPM, new int[]{0,1,2,3}, predInd2CPMIndices, new int[]{2,3,1}, 0, hasIntercept);
        // pred 0, 1 are chosen, add 2
        assertCorrectAddNewPred2CPM(allCPM, new int[]{0,1,2,3}, predInd2CPMIndices, new int[]{0,1}, 2, hasIntercept);
    }

    public static void assertCorrectAddNewPred2CPM(double[][] allCPM, int[] sweepIndices, int[][] predInd2CPMIndices,
                                                   int[] pred2Include, int pred2Add, boolean hasIntercept) {
        // manually generated the cpm after new predictor is included
        int[] allPred = new int[pred2Include.length+1];
        System.arraycopy(pred2Include, 0, allPred, 0, pred2Include.length);
        allPred[pred2Include.length] = pred2Add;
        double[][] manualExtractCPM = extractPredSubsetsCPM(allCPM, allPred, predInd2CPMIndices, hasIntercept);
        sweepCPM(manualExtractCPM, sweepIndices, hasIntercept);
        // addNewPred2CPM generated from program
        double[][] smallCPM = extractPredSubsetsCPM(allCPM, pred2Include, predInd2CPMIndices, hasIntercept);
        sweepCPM(smallCPM, sweepIndices, hasIntercept);
        double[][] subsetCPM = addNewPred2CPM(allCPM, smallCPM, allPred, predInd2CPMIndices, hasIntercept);
        
        // program generated subsetCPM is correct if it equals to smallCPM on part of the matrix that is swept.
        int subsetCPMLastInd = subsetCPM.length-1;
        for (int rInd=0; rInd < sweepIndices.length; rInd++) {
            for (int cInd=0; cInd < sweepIndices.length; cInd++) {
                assertTrue("Expected: "+subsetCPM[rInd][cInd]+" Actual: "+manualExtractCPM[rInd][cInd]+ ". They" +
                        " are different.", Math.abs(subsetCPM[rInd][cInd]-manualExtractCPM[rInd][cInd])<1e-6);
            }
            // compare the last column and row too
            assertTrue("Expected: "+subsetCPM[subsetCPMLastInd][rInd]+" Actual: "+
                    manualExtractCPM[subsetCPMLastInd][rInd]+ ". They are different.", 
                    Math.abs(subsetCPM[subsetCPMLastInd][rInd]-manualExtractCPM[subsetCPMLastInd][rInd])<1e-6);
            assertTrue("Expected: "+subsetCPM[rInd][subsetCPMLastInd]+". Actual: "+
                    manualExtractCPM[rInd][subsetCPMLastInd]+ ". They are different.", 
                    Math.abs(subsetCPM[rInd][subsetCPMLastInd]-manualExtractCPM[rInd][subsetCPMLastInd])<1e-6);

        }
        // for the part of CPM that is not swept, it should equal to the original matrix
        double[][] origCPM = extractPredSubsetsCPM(allCPM, allPred, predInd2CPMIndices, hasIntercept);
        origCPM[subsetCPMLastInd][subsetCPMLastInd] = subsetCPM[subsetCPMLastInd][subsetCPMLastInd];
        for (int rInd=sweepIndices.length; rInd<subsetCPM.length; rInd++) {
            for (int cInd=sweepIndices.length; cInd<subsetCPM.length; cInd++) {
                assertTrue("Expected: "+subsetCPM[rInd][cInd]+". Actual: "+origCPM[rInd][cInd]+".  They are" +
                        " different.", Math.abs(subsetCPM[rInd][cInd]-origCPM[rInd][cInd])<1e-6);
            }
            // check last row/column
            assertTrue("Expected: "+subsetCPM[subsetCPMLastInd][rInd]+" Actual: "+
                            manualExtractCPM[subsetCPMLastInd][rInd]+ ". They are different.",
                    Math.abs(subsetCPM[subsetCPMLastInd][rInd]-origCPM[subsetCPMLastInd][rInd])<1e-6);
            assertTrue("Expected: "+subsetCPM[rInd][subsetCPMLastInd]+". Actual: "+
                            origCPM[rInd][subsetCPMLastInd]+ ". They are different.",
                    Math.abs(subsetCPM[rInd][subsetCPMLastInd]-origCPM[rInd][subsetCPMLastInd])<1e-6);
        }
    }
    
    @Test
    public void testPredIndex2CPMIndices() {
        Scope.enter();
        try {
            Frame origF = Scope.track(parseTestFile("smalldata/glm_test/gaussian_20cols_10000Rows.csv"));
            int[] eCol = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            Arrays.stream(eCol).forEach(x -> origF.replace(x, origF.vec(x).toCategoricalVec()).remove());
            DKV.put(origF);
            Scope.track(origF);
            ModelSelectionModel.ModelSelectionParameters parms = new ModelSelectionModel.ModelSelectionParameters();
            parms._response_column = "C21";
            parms._family = gaussian;
            parms._train = origF._key;
            DataInfo dinfo =  new DataInfo(origF.clone(), null, 1, false,
                    DataInfo.TransformType.STANDARDIZE, DataInfo.TransformType.NONE, true, false,
                    parms.makeImputer(), false, false, false, false, null);
            // manually generate predictor indices to CPM row/column indices
            String[] predictorNames = dinfo._adaptedFrame.names(); // predictor names plus response column
            int numPred = predictorNames.length-1;
            int[] catOffset = dinfo._catOffsets;
            int[] numOffset = dinfo._numOffsets;
            int[][] mPred2CPMMap = new int[numPred][];
            for (int catInd=0; catInd<dinfo._nums; catInd++) {
                int numRowCols = catOffset[catInd+1]-catOffset[catInd];
                mPred2CPMMap[catInd] = IntStream.iterate(catOffset[catInd], x->x+1).limit(numRowCols).toArray();
            }
            int totCol = dinfo._cats+dinfo._nums;
            for (int numInd=0; numInd<dinfo._nums; numInd++)
                mPred2CPMMap[numInd+dinfo._cats] = new int[]{numOffset[numInd]};
            
            // generated from model
            int[][] predictorIndex2CPMIndices = mapPredIndex2CPMIndices(dinfo, parms, predictorNames.length-1);
            
            // compare manually generated and program generated predInd2MapIndices
            assertTrue(mPred2CPMMap.length==predictorIndex2CPMIndices.length);
            for (int index=0; index<numPred; index++) {
                assertArrayEquals("expected and extracted predInd2CPMIndices arrays are not equal", 
                        mPred2CPMMap[index], predictorIndex2CPMIndices[index]);
            }
        } finally {
            Scope.exit();
        }
    }
    
    @Test
    public void testMaxRSweepEnumOnly() {
        Scope.enter();
        try {
            Frame origF = Scope.track(parseTestFile("smalldata/glm_test/gaussian_20cols_10000Rows.csv"));
            int[] eCol = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            Arrays.stream(eCol).forEach(x -> origF.replace(x, origF.vec(x).toCategoricalVec()).remove());
            DKV.put(origF);
            Scope.track(origF);
            ModelSelectionModel.ModelSelectionParameters parms = new ModelSelectionModel.ModelSelectionParameters();
            parms._response_column = "C21";
            parms._ignored_columns = new String[]{"C11","C12","C13","C14","C15","C16","C17","C18","C19","C20"};
            parms._family = gaussian;
            parms._max_predictor_number = 5;
            parms._train = origF._key;
            parms._mode = maxrsweep;
            ModelSelectionModel modelMaxRSweep = new hex.modelselection.ModelSelection(parms).trainModel().get();
            Frame resultFrameSweep = modelMaxRSweep.result();
            Scope.track(resultFrameSweep);
            Scope.track_generic(modelMaxRSweep);
            
            parms._mode = maxr;
            ModelSelectionModel modelMaxR = new hex.modelselection.ModelSelection(parms).trainModel().get();
            Scope.track_generic(modelMaxR);
            Frame resultMaxR = modelMaxR.result();
            Scope.track(resultMaxR);
            TestUtil.assertIdenticalUpToRelTolerance(new Frame(resultFrameSweep.vec(2)), new Frame(resultMaxR.vec(2)), 1e-6);
            TestUtil.assertIdenticalUpToRelTolerance(new Frame(resultFrameSweep.vec(3)), new Frame(resultMaxR.vec(3)), 0);
        } finally {
            Scope.exit();
        }
    }

    @Test
    public void testMaxRSweepMixedColumns() {
        Scope.enter();
        try {
            Frame origF = Scope.track(parseTestFile("smalldata/glm_test/gaussian_20cols_10000Rows.csv"));
            int[] eCol = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            Arrays.stream(eCol).forEach(x -> origF.replace(x, origF.vec(x).toCategoricalVec()).remove());
            DKV.put(origF);
            Scope.track(origF);
            ModelSelectionModel.ModelSelectionParameters parms = new ModelSelectionModel.ModelSelectionParameters();
            parms._response_column = "C21";
            parms._ignored_columns = new String[]{"C11","C12","C13","C14","C15","C16","C17","C18"};
            parms._family = gaussian;
            parms._max_predictor_number = 5;
            parms._train = origF._key;
            parms._mode = maxrsweep;
            ModelSelectionModel modelMaxRSweep = new hex.modelselection.ModelSelection(parms).trainModel().get();
            Frame resultFrameSweep = modelMaxRSweep.result();
            Scope.track(resultFrameSweep);
            Scope.track_generic(modelMaxRSweep);

            parms._mode = maxr;
            ModelSelectionModel modelMaxR = new hex.modelselection.ModelSelection(parms).trainModel().get();
            Scope.track_generic(modelMaxR);
            Frame resultMaxR = modelMaxR.result();
            Scope.track(resultMaxR);
            TestUtil.assertIdenticalUpToRelTolerance(new Frame(resultFrameSweep.vec(2)), new Frame(resultMaxR.vec(2)), 1e-6);
            TestUtil.assertIdenticalUpToRelTolerance(new Frame(resultFrameSweep.vec(3)), new Frame(resultMaxR.vec(3)), 0);
        } finally {
            Scope.exit();
        }
    }
}
