package hex.glm;

import water.*;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.Vec;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.DoubleStream;

import static hex.glm.DispersonTask.CONSTANT_COLUMN_NUMBER;
import static org.apache.commons.math3.special.Gamma.gamma;

/***
 * class to find bounds on infinite series approximation to calculate tweedie dispersion parameter using the 
 * maximum likelihood function in Dunn et.al. in Series evalatuino of Tweedie exponential dispersion model 
 * densities, statistics and computing, Vol 15, 2005.
 */
public class TweedieMLDispersionOnly {
    double _dispersionParameter;   // parameter of optimization
    final double _variancePower;
    int _constNCol;
    int _nWorkingCol = 0;
    double _alpha;
    Frame _infoFrame;   // contains response, mu, weightColumn, constants, max value index, ...
    Frame _mu;
    final String[] _constFrameNames;
    String[] _workFrameNames;
    boolean _weightPresent;
    double _alphaMinus1;
    double _log2;
    double _logDispersionEpsilon;
    double _oneOverDispersion;
    double _logDispersion;
    double _oneMinusAlphaTLogDispersion;
    int _indexBound;    // denotes maximum index we are willing to try
    int _nWVs = 3;
    boolean[] _computationAccuracy; // set to false when upper bound exceeds _indexBound
    double _alphaTimesPI;
    boolean _debugOn;
    
    public TweedieMLDispersionOnly(Frame train, GLMModel.GLMParameters parms, GLMModel model, 
                                   Job job) {
        _variancePower = parms._tweedie_variance_power;
        _dispersionParameter = parms._init_dispersion_parameter;
        _alpha = (2-_variancePower)/(1-_variancePower);
        _alphaMinus1 = _alpha-1;
        _constNCol = CONSTANT_COLUMN_NUMBER;
        _mu = model.score(DKV.<Frame>getGet(parms._train));  // generate prediction
        DKV.put(_mu);
        // form info frame which contains response, mu and weight column if specified
        _infoFrame = formInfoFrame(train, _mu, parms);
        DKV.put(_infoFrame);
        // generate constants used during dispersion parameter update
        DispersonTask.ComputeTweedieConstTsk _tweedieConst = new DispersonTask.ComputeTweedieConstTsk(job, 
                _variancePower, _infoFrame);
        _tweedieConst.doAll(_constNCol, Vec.T_NUM, _infoFrame);
        _constFrameNames = new String[]{"jMaxConst", "zConst", "part2Const", "oneOverY", "oneOverPiY", 
                "firstOrderDerivConst", "secondOrderDerivConst"};
        _infoFrame.add(Scope.track(_tweedieConst.outputFrame(Key.make(), _constFrameNames, null)));
        _debugOn = parms._debugTDispersionOnly;
        if (_debugOn) { // only expand frame when debug is turned on
            _workFrameNames = new String[]{"jOrKMax", "logZ", "_WOrVMax", "dWOrVMax", "d2WOrVMax", "jOrkL", "jOrkU",
                    "djOrkL", "djOrkU", "sumWV", "sumDWV", "sumD2WV", "ll", "dll", "d2ll"};
            _nWorkingCol = _workFrameNames.length;
            Vec[] vecs = _infoFrame.anyVec().makeDoubles(_nWorkingCol, DoubleStream.generate(()
                    -> Math.random()).limit(_nWorkingCol).toArray());
            _infoFrame.add(_workFrameNames, vecs);
            DKV.put(_infoFrame);
        }
        _weightPresent = parms._weights_column != null;
        _log2 = Math.log(2);
        _logDispersionEpsilon = Math.log(parms._tweedie_epsilon);
        _oneOverDispersion = 1.0/_dispersionParameter;
        _logDispersion = Math.log(_dispersionParameter);
        _oneMinusAlphaTLogDispersion = (1-_alpha)*Math.log(_dispersionParameter);
        _indexBound = parms._max_series_index;
        _computationAccuracy = new boolean[_nWVs];
        _alphaTimesPI = _alpha*Math.PI;
    }
    
    public void updatePhiConstant(double dispersionP) {
        _dispersionParameter = dispersionP;
        _oneMinusAlphaTLogDispersion = (1-_alpha)*Math.log(_dispersionParameter);
        _oneOverDispersion = 1.0/_dispersionParameter;
        _logDispersion = Math.log(_dispersionParameter);
    }
    
    public static Frame formInfoFrame(Frame train, Frame mu, GLMModel.GLMParameters parms) {
        Frame infoFrame = new Frame(Key.make());
        String[] colNames;
        Vec[] vecs;
        if (parms._weights_column != null) {
            colNames = new String[]{parms._response_column, mu.names()[0], parms._weights_column};
            vecs = new Vec[]{train.vec(parms._response_column), mu.vec(0), train.vec(parms._weights_column)};
        } else {
            colNames = new String[]{parms._response_column, mu.names()[0]};
            vecs = new Vec[]{train.vec(parms._response_column), mu.vec(0)};
        }
        infoFrame.add(colNames, vecs);
        return infoFrame;
    }
    
    
    public void cleanUp() {
        DKV.remove(_mu._key);
        DKV.remove(_infoFrame._key);
    }
}
