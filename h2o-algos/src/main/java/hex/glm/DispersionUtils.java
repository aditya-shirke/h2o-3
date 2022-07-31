package hex.glm;

import hex.DataInfo;
import water.Job;
import water.fvec.Frame;
import water.util.Log;

public class DispersionUtils {
    /***
     * Estimate dispersion factor using maximum likelihood.  I followed section IV of the doc in 
     * https://h2oai.atlassian.net/browse/PUBDEV-8683 . 
     */
    public static double estimateGammaMLSE(GLMTask.ComputeGammaMLSETsk mlCT, double seOld, double[] beta, 
                                           GLMModel.GLMParameters parms, ComputationState state, Job job, GLMModel model) {
        double constantValue = mlCT._wsum + mlCT._sumlnyiOui - mlCT._sumyiOverui;
        DataInfo dinfo = state.activeData();
        Frame adaptedF = dinfo._adaptedFrame;
        long currTime = System.currentTimeMillis();
        long modelBuiltTime = currTime - model._output._start_time;
        long timeLeft = parms._max_runtime_secs > 0 ? (long) (parms._max_runtime_secs * 1000 - modelBuiltTime) : Long.MAX_VALUE;

        // stopping condition for while loop are:
        // 1. magnitude of iterative change to se < EPS
        // 2. there are more than MAXITERATIONS of updates
        // 2. for every 100th iteration, we check for additional stopping condition:
        //    a.  User requests stop via stop_requested;
        //    b.  User sets max_runtime_sec and that time has been exceeded.
        for (int index=0; index<parms._max_iterations_dispersion; index++) {
            GLMTask.ComputeDiTriGammaTsk ditrigammatsk = new GLMTask.ComputeDiTriGammaTsk(null, dinfo, job._key, beta,
                    parms, seOld).doAll(adaptedF);
            double numerator = mlCT._wsum*Math.log(seOld)-ditrigammatsk._sumDigamma+constantValue; // equation 2 of doc
            double denominator = mlCT._wsum/seOld - ditrigammatsk._sumTrigamma;  // equation 3 of doc
            double change = numerator/denominator;
            if (denominator == 0 || Double.isNaN(change))
                return seOld;
            if (Math.abs(change) < parms._dispersion_epsilon) // stop if magnitude of iterative updates to se < EPS
                return seOld-change;
            else {
                double se = seOld - change;
                if (se < 0) // heuristic to prevent seInit <= 0
                    seOld *= 0.5;
                else
                    seOld = se;
            }

            if ((index % 100 == 0) && // check for additional stopping conditions for every 100th iterative steps
                    (job.stop_requested() ||  // user requested stop via stop_requested()
                            (System.currentTimeMillis()-currTime) > timeLeft)) { // time taken to find dispersino exceeds GLM model building time
                Log.warn("gamma dispersion parameter estimation was interrupted by user or due to time out.  Estimation " +
                        "process has not converged. Increase your max_runtime_secs if you have set maximum runtime for your " +
                        "model building process.");
                return seOld;
            }
        }
        Log.warn("gamma dispersion parameter estimation fails to converge within "+
                parms._max_iterations_dispersion+" iterations.  Increase max_iterations_dispersion or decrease " +
                "dispersion_epsilon.");
        return seOld;
    }

    public static double estimateTweedieDispersionOnly(GLMModel.GLMParameters parms, GLMModel model, Job job) {
        long currTime = System.currentTimeMillis();
        long modelBuiltTime = currTime - model._output._start_time;
        long timeLeft = parms._max_runtime_secs > 0 ? (long) (parms._max_runtime_secs * 1000 - modelBuiltTime) 
                : Long.MAX_VALUE;

        TweedieMLDispersionOnly tDispersion = new TweedieMLDispersionOnly(parms.train(), parms, model, job);
        DispersonTask.ComputeMaxSumSeriesTsk computeTask = new DispersonTask.ComputeMaxSumSeriesTsk(job, tDispersion,
                parms);

        double seOld = tDispersion._dispersionParameter;   // initial value of dispersion parameter
        double change, se, numerator, denominator;
        for (int index=0; index<parms._max_iterations_dispersion; index++) {
            computeTask.doAll(tDispersion._infoFrame);
            // set new alpha
            numerator = computeTask._dLogLL;
            denominator = computeTask._d2LogLL;
            change = numerator/denominator; // no line search is employed at the moment ToDo: add line search
            if (denominator == 0 || Double.isNaN(change))
                return seOld;
            if (Math.abs(change) < parms._dispersion_epsilon) {
                tDispersion.cleanUp();
                return seOld - change;
            } else {
                se = seOld - change;
                if (se < 0)
                    seOld *= 0.5;
                else
                    seOld = se;
            }
            computeTask.updatePhiParams(se);
            // set step size ??
            if ((index % 100 == 0) && // check for additional stopping conditions for every 100th iterative steps
                    (job.stop_requested() ||  // user requested stop via stop_requested()
                            (System.currentTimeMillis()-currTime) > timeLeft)) { // time taken exceeds model build time
                Log.warn("tweedie dispersion parameter estimation was interrupted by user or due to time out." +
                        "  Estimation process has not converged. Increase your max_runtime_secs if you have set " +
                        "maximum runtime for your model building process.");
                tDispersion.cleanUp();
                return seOld;
            }
        }
        tDispersion.cleanUp();
        return tDispersion._dispersionParameter;
    }
}
