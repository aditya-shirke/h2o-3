setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

# check we can get added and removed predictors in the result frame and call from model
testAddedRemovedPreds <- function() {
  bhexFV <- h2o.uploadFile(locate("smalldata/logreg/prostate.csv"))
  Y <- "GLEASON"
  X <- c("AGE","RACE","CAPSULE","DCAPS","PSA","VOL","DPROS")
  Log.info("Build the MaxRGLM model")
  numModel <- 7
  allsubsetsModel <- h2o.modelSelection(y=Y, x=X, seed=12345, training_frame = bhexFV, max_predictor_number=numModel, mode="allsubsets")
  assertAddedRemovedPreds(allsubsetsModel)
  maxrModel <- h2o.modelSelection(y=Y, x=X, seed=12345, training_frame = bhexFV, max_predictor_number=numModel, mode="maxr")
  assertAddedRemovedPreds(marxModel)
  maxrsweepModel <- h2o.modelSelection(y=Y, x=X, seed=12345, training_frame = bhexFV, max_predictor_number=numModel, mode="maxrsweep")
  assertAddedRemovedPreds(marxsweepModel)
  backwardModel <- h2o.modelSelection(y=Y, x=X, seed=12345, training_frame = bhexFV, mode="backward")
  assertAddedRemovedPreds(backwardModel)
}

assertAddedRemovedPreds <- function(model) {
  browser()
  resultFrame <- h2o.result(model)
  removedPF <- resultFrame[5]
  removedPreds <- h2o.get_predictors_removed_per_step(model)
  assertFrameListEquals(removedPF, removedPreds)
  
  if (model@parameters$mode != 'backward') {
    addPreds <- h2o.get_predictors_added_per_step(model)
    addPredsF <- resultFrame[6]
    assertFrameListEquals(addPredsF, addPreds)
  }
}

assertFrameListEquals <- function(colFrame, oneList) {
  numEle <- h2o.nrow(colFrame)
  for (ind in c(1:numEle)) {
    if (colFrame[ind, 1] == "") {
      expect_true(is.null(oneList[ind]))
    } else {
      expect_true(oneList[ind] == colFrame[ind,1])
    }
  }
}

doTest("ModelSelection: test added and removed predictors in result frame", testAddedRemovedPreds)
