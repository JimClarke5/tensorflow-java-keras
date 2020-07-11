/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.metrics.impl;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.tensorflow.DataType;
import org.tensorflow.keras.metrics.*;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.backend.tf.ControlDependencies;
import org.tensorflow.keras.backend.tf.Tuple;
import org.tensorflow.keras.losses.impl.LossesImpl;
import static org.tensorflow.keras.losses.impl.LossesImpl.l2Normalize;
import org.tensorflow.keras.utils.ShapeUtils;
import org.tensorflow.keras.utils.SymbolicShape;
import org.tensorflow.keras.utils.SymbolicShapeDict;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.OneHot;
import org.tensorflow.op.core.ReduceSum;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Equal;
import org.tensorflow.op.nn.TopK;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

/**
 *
 * @author Jim Clarke
 */
public class MetricsImpl {

    public static final float NEG_INF = -1e10f;
    public static final int DEFAULT_K = 5;
    
    
    private static final String DEFAULT_NAME = "MetricsImpl%d";
    private static final AtomicInteger counter = new AtomicInteger();
    
    // used for generating unique metric names
    private static final String genUniqueName() {
        return String.format(DEFAULT_NAME, counter.getAndIncrement());
    }

    public static Operand accuracy(Ops tf, Operand yTrue, Operand yPred) {
        Accuracy instance = new Accuracy(tf, genUniqueName());
        Operand result = instance.callOnce( yTrue, yPred);
        return result;

    }
    
    public static Operand accuracy(Ops tf, Operand yTrue, Operand yPred, Operand sampleWeight) {
        Accuracy instance = new Accuracy(tf, genUniqueName());
        Operand result = instance.callOnce( yTrue, yPred, sampleWeight);
        return result;

    }

    /**
     * Calculates how often predictions matches binary labels.
     *
     * @param tf The TensorFlowOps
     * @param yTrue Ground truth values.
     * @param yPred The predicted values
     * @param threshold he threshold for deciding whether prediction values are
     * 1 or 0
     * @return Binary accuracy values
     */
    public static Operand binary_accuracy(Ops tf, Operand yTrue, Operand yPred) {
        BinaryAccuracy instance = new BinaryAccuracy(tf, genUniqueName());
        return instance.callOnce( yTrue, yPred);
    }
    
    public static Operand binary_accuracy(Ops tf, Operand yTrue, Operand yPred, float threshold) {
        BinaryAccuracy instance = new BinaryAccuracy(tf, genUniqueName(), threshold);
        return instance.callOnce( yTrue, yPred);
    }
    
    /**
     * Calculates how often predictions matches binary labels.
     *
     * @param tf The TensorFlowOps
     * @param yTrue Ground truth values.
     * @param yPred The predicted values
     * @param threshold he threshold for deciding whether prediction values are
     * 1 or 0
     * @return Binary accuracy values
     */
    public static Operand binary_accuracy(Ops tf, Operand yTrue, Operand yPred, Operand sampleWeight) {
        BinaryAccuracy instance = new BinaryAccuracy(tf, genUniqueName());
        System.out.println("binary_accuracy: " + instance.getThreshold());
        return instance.callOnce( yTrue, yPred, sampleWeight);
    }
    
    public static Operand binary_accuracy(Ops tf, Operand yTrue, Operand yPred, Operand sampleWeight, float threshold) {
        BinaryAccuracy instance = new BinaryAccuracy(tf, genUniqueName(), threshold);
        System.out.println("binary_accuracy: " + instance.getThreshold());
        return instance.callOnce( yTrue, yPred, sampleWeight);
    }

    public static Operand categorical_accuracy(Ops tf, Operand yTrue, Operand yPred) {
       CategoricalAccuracy instance = new CategoricalAccuracy(tf, genUniqueName());
        return instance.callOnce( yTrue, yPred);
    }
    
    public static Operand categorical_accuracy(Ops tf, Operand yTrue, Operand yPred, Operand sampleWeight) {
       CategoricalAccuracy instance = new CategoricalAccuracy(tf, genUniqueName());
        return instance.callOnce( yTrue, yPred, sampleWeight);
    }

    /**
     * Returns op to update the given confusion matrix variables. For every pair
     * of values in y_true and y_pred: true_positive: y_true == True and y_pred
     * > thresholds false_negatives: y_true == True and y_pred <= thresholds
     * true_negatives: y_true == False and y_pred <= thresholds
     * false_positive: y_true == False and y_pred > thresholds The results will
     * be weighted and added together. When multiple thresholds are provided, we
     * will repeat the same for every threshold. For estimation of these metrics
     * over a stream of data, the function creates an ` * update_op` operation
     * that updates the given variables. If `sample_weight` is `None`, weights
     * default to 1. Use weights of 0 to mask values.
     *
     *
     * @param confusion_matrix_cond
     * @param accumulator
     * @param yTrue A `Tensor` whose shape matches `y_pred`. Will be cast to
     * `bool`.
     * @param yPred `Tensor` of arbitrary shape and whose values are in the
     * range `[0, 1]`.
     * @param thresholds
     * @param sampleWeights
     * @return an update op
     */
    static Op update_confusion_matrix_variables(Ops tf, ConfusionMatrixEnum confusion_matrix_cond,
            Variable<TFloat32> accumulator, Operand yTrue, Operand yPred,
            float[] thresholds, Operand sampleWeights) {

        yTrue = tf.dtypes.cast(yTrue, TFloat32.DTYPE);
        yPred = tf.dtypes.cast(yPred, TFloat32.DTYPE);
        Operand one_thresh = tf.constant(true);

        List<Op> updateOperations = new ArrayList<>();

        tf.assertThat(
                tf.math.greaterEqual(yPred, K.zero(tf, yPred.asOutput().dataType())),
                (Iterable<Operand<?>>) Arrays.asList(tf.constant("predictions must be >= 0")).iterator()
        );
        tf.assertThat(
                tf.math.lessEqual(yPred, K.one(tf, yPred.asOutput().dataType())),
                (Iterable<Operand<?>>) Arrays.asList(tf.constant("predictions must be <= 1")).iterator()
        );
        Tuple ops = LossesImpl.squeezeOrExpandDimensions(tf, yPred, yTrue, sampleWeights);
        yPred = ops.getPredictions();
        yTrue = ops.getLabels();
        sampleWeights = ops.getSampleWeights();
        Shape trueShape = yTrue.asOutput().shape();
        Shape predShape = yPred.asOutput().shape();
        assert ShapeUtils.isCompatibleWith(predShape, trueShape) :
                String.format("Prediction shape  %s is not compatible with Labels shape %s",
                        predShape, trueShape);

        long num_predictions = predShape.size(0);
        Operand numLabels;
        if (predShape.numDimensions() == 1) {
            numLabels = tf.constant(1);
        } else {
            numLabels = tf.prod(tf.constant(predShape.tail()), tf.constant(0));
        }
        Operand threshLabelTile = tf.select(one_thresh, numLabels, tf.constant(1));

        // TODO multi-label
        Operand predictionsExtraDim = tf.reshape(yPred, tf.constant(new long[]{1, -1}));
        //thresh_tiles = [1, num_predictions * numLabels]
        //data_tiles = [num_thresholds, 1]
        // TODO
        return null;

    }

    public static List<Op> assert_shapes(Ops tf, List<SymbolicShape> symbols, String message) {
        List<Op> updateOperations = new ArrayList<>();
        symbols.forEach(symbol -> {
            Operand operand = symbol.getOperand();
            int rank = symbol.rank();
            Operand tfRank = tf.rank(operand);
            Op assertion = tf.assertThat(
                    tf.math.equal(tfRank, tf.constant(rank)),
                    Arrays.asList(tf.constant(message)));
            updateOperations.add(assertion);

        });

        SymbolicShapeDict dict = new SymbolicShapeDict();

        symbols.forEach(symbol -> {
            AtomicLong ll = new AtomicLong();
            symbol.getSymbols().forEach(s -> {
                Long size = dict.get(s);
                if (size == null) {
                    size = symbol.getOperand().asOutput().shape().size((int) ll.get());
                    dict.put(s, size);
                }
                Op assertion = tf.assertThat(
                        tf.math.equal(
                                tf.shape.size(symbol.getOperand(), tf.constant(ll.getAndIncrement()), TInt64.DTYPE),
                                tf.constant(size)),
                        Arrays.asList(tf.constant(message)));
                updateOperations.add(assertion);

            });
        });

        return updateOperations;
    }

    public static List<Op> update_confusion_matrix_variables(Ops tf,
            Map<ConfusionMatrixEnum, Variable> variablesToUpdate,
            Map<ConfusionMatrixEnum, Assign> varInitalizers,
            Operand yTrue, Operand yPred,
            float[] thresholds, Integer topK, Integer classId, Operand sampleWeight, boolean multiLabel, Operand labelWeights) {
        assert !(multiLabel && labelWeights != null) :
                "`labelWeights` for multilabel data should be handled outside of `update_confusion_matrix_variables` when `multiLabel` is True.'";
        if (variablesToUpdate == null || variablesToUpdate.isEmpty()) {
            return Collections.EMPTY_LIST;
        }
        //TODO remove 
        debug("beforeCast/yTrue", yTrue);
        //TODO remove 
        debug("beforeCast/yPred", yPred);
        yTrue = tf.dtypes.cast(yTrue, TFloat32.DTYPE);
        yPred = tf.dtypes.cast(yPred, TFloat32.DTYPE);
        //TODO remove 
        debug("afterCast/yTrue", yTrue);
        //TODO remove 
        debug("afterCast/yPred", yPred);
        Operand<TInt32> numThresholds;
        Operand<TBool> oneThresh;
        if (multiLabel) {
            numThresholds = tf.shape.size(yTrue, tf.constant(0));
            oneThresh = tf.math.equal(tf.constant(1), tf.constant(thresholds.length));
        } else {
            // TODO handle Ragged Tensors????
            // [y_pred,
            //    y_true], _ = ragged_assert_compatible_and_get_flat_values([y_pred, y_true],
            //                                                   sample_weight)
            numThresholds = tf.constant(thresholds.length);
            oneThresh = tf.constant(true);
        }

        List<Op> controlOps = new ArrayList<>();
        controlOps.add(tf.assertThat(
                tf.math.greaterEqual(yPred, K.zero(tf, yPred.asOutput().dataType())),
                Arrays.asList(tf.constant("predictions must be >= 0"))
        ));
        controlOps.add(tf.assertThat(
                tf.math.lessEqual(yPred, K.one(tf, yPred.asOutput().dataType())),
                Arrays.asList(tf.constant("predictions must be <= 1"))
        ));
        Tuple result = LossesImpl.squeezeOrExpandDimensions(tf, yTrue, yPred, sampleWeight);
        yPred = result.getPredictions();
        //TODO remove 
        debug("yPred2", yPred);
        yTrue = result.getLabels();
        sampleWeight = result.getSampleWeights();
        //TODO remove 
        debug("sampleWeight", sampleWeight);
        assert ShapeUtils.isCompatibleWith(yPred.asOutput().shape(), yTrue.asOutput().shape()) :
                String.format("Shapes %s and %s are incompatible)", yPred.asOutput().shape().toString(),
                        yTrue.asOutput().shape().toString());

        if (topK != null) {
            yPred = filterTopK(tf, yPred, topK);
            //TODO remove 
            MetricsImpl.debug("topK/yPred", yPred);
        }

        if (classId != null) {
            yTrue = tf.squeeze(tf.gather(yTrue, tf.constant(new int[]{classId}), tf.constant(1)));
            yPred = tf.squeeze(tf.gather(yPred, tf.constant(new int[]{classId}), tf.constant(1)));
            yTrue = tf.expandDims(yTrue, tf.constant(0));
            yPred = tf.expandDims(yPred, tf.constant(0));
            //TODO remove 
            debug("classId/yTrue", yTrue);
            //TODO remove 
            debug("classId/yPred", yPred);
        }
        org.tensorflow.op.core.Shape<TInt32> predShape = tf.shape(yPred);
        Operand<TInt32> numPredictions = tf.reshape(tf.shape.size(yPred, tf.constant(0)), tf.constant(Shape.scalar()));
        //TODO remove 
        //TODO remove debug("predShape", predShape);
        Operand<TInt32> numLabels = tf.select(
                tf.math.equal(tf.shape.numDimensions(predShape), tf.constant(1)),
                tf.constant(1),
                tf.reduceProd(tf.shape.takeLast(predShape,
                        tf.math.sub(tf.shape.numDimensions(predShape),
                                tf.constant(1))),
                        tf.constant(0))
        );
        //TODO remove  
        debug("numPredictions", numPredictions);
        //TODO remove  
        debug("numLabels", numLabels);
        Operand<TInt32> threshLabelTile = tf.select(
                oneThresh, numLabels, tf.constant(1));

        Operand predictionsExtraDim;
        Operand labelsExtraDim;
        if (multiLabel) {
            predictionsExtraDim = tf.expandDims(yPred, tf.constant(0));
            labelsExtraDim = tf.expandDims(
                    tf.dtypes.cast(yTrue, TBool.DTYPE), tf.constant(0));
        } else {
            //TODO remove  
            debug("yPred", yPred);
            predictionsExtraDim = tf.reshape(yPred, tf.constant(Shape.of(1, -1)));
            //TODO remove  
            debug("predictionsExtraDim", predictionsExtraDim);
            labelsExtraDim = tf.reshape(
                    tf.dtypes.cast(yTrue, TBool.DTYPE),
                    tf.constant(Shape.of(1, -1)));
        }
        //TODO remove 
        debug("labelsExtraDim", labelsExtraDim);

        List<Operand<TInt32>> threshPretileShape;
        List<Operand<TInt32>> threshTiles;
        List<Operand<TInt32>> dataTiles;
        if (multiLabel) {
            threshPretileShape = Arrays.asList(numThresholds, tf.constant(1), tf.constant(-1));

            threshTiles = Arrays.asList(tf.constant(1), numPredictions,
                    threshLabelTile);
            dataTiles = Arrays.asList(
                    numThresholds, tf.constant(1), tf.constant(1));
        } else {
            threshPretileShape = Arrays.asList(numThresholds, tf.constant(-1));
            Operand mul = tf.math.mul(numPredictions, numLabels);
            threshTiles = Arrays.asList(tf.constant(1), mul);
            dataTiles = Arrays.asList(numThresholds, tf.constant(1));
        }

        Operand thresholdsReshaped = tf.reshape(tf.constant(thresholds), tf.stack(threshPretileShape));
        Operand threshTilesShape = tf.stack(threshTiles);
        Operand threshTiled = tf.tile(thresholdsReshaped, threshTilesShape);
        //TODO remove 
        debug("threshTiled", threshTiled);
        Operand predsTiled = tf.tile(predictionsExtraDim, tf.stack(dataTiles));
        //TODO remove 
        debug("predsTiled", predsTiled);

        //Compare predictions and threshold.
        Operand predIsPos = tf.math.greater(predsTiled, threshTiled);
        // Tile labels by number of thresholds
        Operand labelIsPos = tf.tile(labelsExtraDim, tf.stack(dataTiles));
        //TODO remove 
        debug("predIsPos", predIsPos);
        //TODO remove 
        debug("labelIsPos", labelIsPos);
        Operand weightsTiled;
        if (sampleWeight != null) {
            sampleWeight = tf.broadcastTo(tf.dtypes.cast(sampleWeight, TFloat32.DTYPE), tf.shape(yPred));
            //TODO remove debug("sampleWeight_broadcast", sampleWeight);
            weightsTiled = tf.tile(
                    tf.reshape(sampleWeight, tf.stack(threshTiles)),
                    tf.stack(dataTiles));
            //TODO remove debug("weightsTiled1", weightsTiled);
        } else {
            weightsTiled = null;
        }

        if (labelWeights != null && !multiLabel) {
            labelWeights = tf.expandDims(tf.identity(labelWeights), tf.constant(0));
            labelWeights = tf.broadcastTo(
                    tf.dtypes.cast(labelWeights, TFloat32.DTYPE), yPred);
            Operand labelWeightsTiled = tf.tile(
                    tf.reshape(labelWeights, tf.stack(threshTiles)),
                    tf.stack(dataTiles));
            if (weightsTiled == null) {
                weightsTiled = labelWeightsTiled;
                //TODO remove  debug("weightsTiled_labelWeightsTiled", weightsTiled);
            } else {
                weightsTiled = tf.math.mul(weightsTiled, labelWeightsTiled);
                //TODO remove  debug("weightsTiled_mul", weightsTiled);
            }
        }

        List<Op> updateOps = new ArrayList<>();
        Map<ConfusionMatrixEnum, Operand[]> loopVars = new HashMap<>();
        loopVars.put(ConfusionMatrixEnum.TRUE_POSITIVES, new Operand[]{labelIsPos, predIsPos});
        Variable update_tn = variablesToUpdate.get(ConfusionMatrixEnum.TRUE_NEGATIVES);
        Variable update_fp = variablesToUpdate.get(ConfusionMatrixEnum.FALSE_POSITIVES);
        Variable update_fn = variablesToUpdate.get(ConfusionMatrixEnum.FALSE_NEGATIVES);

        Operand predIsNeg = null;
        Operand labelIsNeg;
        if (update_fn != null || update_tn != null) {
            predIsNeg = tf.math.logicalNot(predIsPos);
            //TODO remove 
            debug("predIsNeg", predIsNeg);
            loopVars.put(ConfusionMatrixEnum.FALSE_NEGATIVES, new Operand[]{labelIsPos, predIsNeg});
        }

        if (update_fp != null || update_tn != null) {
            labelIsNeg = tf.math.logicalNot(labelIsPos);
            //TODO remove 
            debug("labelIsNeg", labelIsNeg);
            loopVars.put(ConfusionMatrixEnum.FALSE_POSITIVES, new Operand[]{labelIsNeg, predIsPos});
            if (update_tn != null) {
                loopVars.put(ConfusionMatrixEnum.TRUE_NEGATIVES, new Operand[]{labelIsNeg, predIsNeg});
            }
        }

        final Operand weightsTiledF = weightsTiled;
        loopVars.keySet().forEach((c) -> {
            if (variablesToUpdate.containsKey(c)) {
                Operand[] op = loopVars.get(c);
                // op[0] = label, op[1] == prediction
                updateOps.add(
                        weightedAssignAdd(tf, c, op[0], op[1], weightsTiledF, variablesToUpdate.get(c), varInitalizers.get(c)));
            }
        });

        return updateOps;

    }

    private static Operand weightedAssignAdd(Ops tf, ConfusionMatrixEnum c, Operand label,
            Operand pred, Operand weights, Variable variable, Assign initializer) {

        //TODO remove 
        MetricsImpl.debug("weightedAssignAdd/" + c + "/label", label);
        //TODO remove 
        MetricsImpl.debug("weightedAssignAdd/" + c + "/pred", pred);
        Operand label_and_pred = tf.dtypes.cast(tf.math.logicalAnd(label, pred), TFloat32.DTYPE);

        if (weights != null) {
            label_and_pred = tf.math.mul(label_and_pred, weights);
        }
        //TODO remove  
        MetricsImpl.debug("weightedAssignAdd/" + c + "/label_and_pred", label_and_pred);
        Operand valueSum = tf.reduceSum(label_and_pred, tf.constant(1));
        //TODO remove 
        MetricsImpl.debug("weightedAssignAdd/" + c + "/valueSum", valueSum);
        Operand assignAdd;
        if (initializer != null) {
            assignAdd = ControlDependencies.addControlDependencies(tf,
                    tfc -> tfc.assignAdd(variable, valueSum), "weightedAssignAdd", initializer);
        } else {
            assignAdd = tf.assignAdd(variable, valueSum);
        }
        return assignAdd;
    }

    private static Operand filterTopK(Ops tf, Operand x, int topK) {
        DataType dtype = x.asOutput().dataType();
        Shape xShape = x.asOutput().shape();
        TopK top = tf.nn.topK(x, tf.constant(topK), TopK.sorted(false));
        OneHot oneHot = tf.oneHot(
                top.indices(),
                tf.dtypes.cast(tf.constant(xShape.size(xShape.numDimensions() - 1)), TInt32.DTYPE),
                tf.constant(1),
                tf.constant(0),
                OneHot.axis(-1L));
        Operand topKMask = tf.reduceSum(
                oneHot,
                tf.constant(-2));

        //x * top_k_mask + NEG_INF * (1 - top_k_mask)
        topKMask = tf.dtypes.cast(topKMask, dtype);
        Operand add1 = tf.math.mul(x, topKMask);
        Operand add2 = tf.math.mul(
                tf.constant(NEG_INF),
                tf.math.sub(tf.dtypes.cast(tf.constant(1), dtype), topKMask));
        Operand result = tf.math.add(add1, add2);
        return result;
    }

    /**
     * y
     * Computes Kullback-Leibler divergence loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     *
     * @return the loss
     */
    public static Operand KLD(Ops tf, Operand yTrue, Operand yPred) {
        return kullback_leibler_divergence(tf, yTrue, yPred);
    }

    /**
     * Computes Kullback-Leibler divergence loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand kld(Ops tf, Operand yTrue, Operand yPred) {
        return kullback_leibler_divergence(tf, yTrue, yPred);
    }

    /**
     * Computes Kullback-Leibler divergence loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand kullback_leibler_divergence(Ops tf, Operand yTrue, Operand yPred) {
        KLDivergence instance = new KLDivergence(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }
    
     /**
     * Computes Kullback-Leibler divergence loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand logCoshError(Ops tf, Operand yTrue, Operand yPred) {
        LogCoshError instance = new LogCoshError(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the mean absolute error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand MAE(Ops tf, Operand yTrue, Operand yPred) {
        return mean_absolute_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mae(Ops tf, Operand yTrue, Operand yPred) {
        return mean_absolute_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mean_absolute_error(Ops tf, Operand yTrue, Operand yPred) {
        MeanAbsoluteError instance = new MeanAbsoluteError(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the mean absolute percentage error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue
     * @param yPred
     * @return the loss
     */
    public static Operand MAPE(Ops tf, Operand yTrue, Operand yPred) {
        return mean_absolute_percentage_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute percentage error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mape(Ops tf, Operand yTrue, Operand yPred) {
        return mean_absolute_percentage_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean absolute percentage error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mean_absolute_percentage_error(Ops tf, Operand yTrue, Operand yPred) {
        MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the mean squared error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand MSE(Ops tf, Operand yTrue, Operand yPred) {
        return mean_squared_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mse(Ops tf, Operand yTrue, Operand yPred) {
        return mean_squared_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared error between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mean_squared_error(Ops tf, Operand yTrue, Operand yPred) {
        MeanSquaredError instance = new MeanSquaredError(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the mean squared logarithmic error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand MSLE(Ops tf, Operand yTrue, Operand yPred) {
        return mean_squared_logarithmic_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared logarithmic error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand msle(Ops tf, Operand yTrue, Operand yPred) {
        return mean_squared_logarithmic_error(tf, yTrue, yPred);
    }

    /**
     * Computes the mean squared logarithmic error between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand mean_squared_logarithmic_error(Ops tf, Operand yTrue, Operand yPred) {
        MeanSquaredLogarithmicError instance = new MeanSquaredLogarithmicError(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the binary crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred the predictions
     * @return the loss
     */
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred) {
        return binary_crossentropy(tf, yTrue, yPred, false, 0.0F);

    }

    /**
     * Computes the binary crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred the predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @return the loss
     */
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits) {
        return binary_crossentropy(tf, yTrue, yPred, fromLogits, 0.0F);
    }

    /**
     * Computes the binary crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred the predictions
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @return the loss
     */
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred, float labelSmoothing) {
        return binary_crossentropy(tf, yTrue, yPred, false, labelSmoothing);
    }

    /**
     * Computes the binary crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred the predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @return the loss
     */
    public static Operand binary_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits, float labelSmoothing) {
        BinaryCrossentropy instance = new BinaryCrossentropy(tf, genUniqueName(), fromLogits, labelSmoothing);
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred) {
        return categorical_crossentropy(tf, yTrue, yPred, false, 0.0F);
    }

    /**
     * Computes the categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @return the loss
     */
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits) {
        return categorical_crossentropy(tf, yTrue, yPred, fromLogits, 0.0F);
    }

    /**
     * Computes the categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @return the loss
     */
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, float labelSmoothing) {
        return categorical_crossentropy(tf, yTrue, yPred, false, labelSmoothing);
    }

    /**
     * Computes the categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @return the loss
     */
    public static Operand categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits, float labelSmoothing) {
        CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, genUniqueName(), fromLogits, labelSmoothing);
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the categorical hinge loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand categorical_hinge(Ops tf, Operand yTrue, Operand yPred) {
        CategoricalHinge instance = new CategoricalHinge(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the cosine similarity between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand cosine_similarity(Ops tf, Operand yTrue, Operand yPred) {
        CosineSimilarity instance = new CosineSimilarity(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the cosine similarity between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand cosine_proximity(Ops tf, Operand yTrue, Operand yPred) {
        return cosine_proximity(tf, yTrue, yPred, -1);
    }

    /**
     * Computes the cosine similarity between labels and predictions.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param axis The dimension along which the cosine similarity is computed.
     * @return the loss
     */
    public static Operand cosine_proximity(Ops tf, Operand yTrue, Operand yPred, int axis) {
        yTrue = l2Normalize(tf, yTrue, axis);
        yPred = l2Normalize(tf, yPred, axis);
        Operand mathMul = tf.math.mul(yTrue, yPred);
        Operand sum = tf.reduceSum(mathMul, tf.constant(axis), ReduceSum.keepDims(Boolean.FALSE));
        return sum;
    }

    /**
     * Computes the hinge loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand hinge(Ops tf, Operand yTrue, Operand yPred) {
        Hinge instance = new Hinge(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the Poisson loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand poisson(Ops tf, Operand yTrue, Operand yPred) {
        Poisson instance = new Poisson(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Computes the sparse categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param axis The dimension along which the entropy is computed.
     * @return the loss
     */
    public static Operand sparse_categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits) {
        return sparse_categorical_crossentropy(tf, yTrue, yPred, fromLogits, -1);
    }

    /**
     * Computes the sparse categorical crossentropy loss.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param axis The dimension along which the entropy is computed.
     * @return the loss
     */
    public static Operand sparse_categorical_crossentropy(Ops tf, Operand yTrue, Operand yPred, boolean fromLogits, int axis) {
        SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, genUniqueName(), fromLogits, axis);
        return instance.callOnce(yTrue, yPred);
    }

    /**
     * Calculates how often predictions matches integer labels.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return Sparse categorical accuracy values.
     */
    public static Operand sparse_categorical_accuracy(Ops tf, Operand yTrue, Operand yPred) {
        Operand yPredRank = tf.rank(yPred);
        Operand yTrueRank = tf.rank(yTrue);
        
        yTrue = tf.squeeze(yTrue);
        yPred = tf.math.argMax(yPred, tf.constant(-1));
        
        if(yPred.asOutput().dataType().equals(yTrue.asOutput().dataType())) {
            yPred = tf.dtypes.cast(yPred, yTrue.asOutput().dataType());
        }
        
        return tf.dtypes.cast(tf.math.equal(yTrue, yPred), TFloat32.DTYPE);
    }

    /**
     * Computes the squared hinge loss between y_true and y_pred.
     *
     * @param tf the TensorFlow Ops
     * @param yTrue true targets
     * @param yPred predictions
     * @return the loss
     */
    public static Operand squared_hinge(Ops tf, Operand yTrue, Operand yPred) {
        SquaredHinge instance = new SquaredHinge(tf, genUniqueName());
        return instance.callOnce(yTrue, yPred);
    }

    public static <T extends TNumber> Operand<TFloat32> top_k_categorical_accuracy(
            Ops tf, Operand<T> labels, Operand<T> predictions, int k) {
        Operand<TFloat32> castPredictions = tf.dtypes.cast(predictions, TFloat32.DTYPE);
        return tf.dtypes.cast(
                tf.nn.inTopK(castPredictions, tf.math.argMax(labels, tf.constant(-1)), tf.constant((long) k)),
                TFloat32.DTYPE);
    }

    public static <T extends TNumber> Operand<TFloat32> sparse_top_k_categorical_accuracy(
            Ops tf, Operand<T> labels, Operand<T> predictions, int k) {
        int predictionsRank = predictions.asOutput().shape().numDimensions();
        int labelsRank = labels.asOutput().shape().numDimensions();

        if (predictionsRank != -1 && labelsRank != -1) {
            if (predictionsRank > 2) {
                predictions = tf.shape.reduceDims(predictions, tf.constant(1));
            }
            if (labelsRank > 1) {
                labels = tf.shape.flatten(labels);
            }
        }
        Operand<TFloat32> castPredictions = tf.dtypes.cast(predictions, TFloat32.DTYPE);
        return tf.dtypes.cast(
                tf.nn.inTopK(castPredictions, tf.dtypes.cast(labels, TInt32.DTYPE), tf.constant(k)),
                TFloat32.DTYPE);
    }
    
    
    // helper functions
    private static void initialize(Graph graph) {
        if (graph != null) {
            try (Session session = new Session(graph)) {
                for (Op initializer : graph.initializers()) {
                    session.runner().addTarget(initializer).run();
                }
            }
        }
    }

    private static void run(Graph graph, Op op) {
        if (graph != null) {
            try (Session session = new Session(graph)) {
                session.run(op);
            }
        }
    }

    private static <T extends TType> void print(Ops tf, String prefix, Operand<T> operand) {
        Graph graph = (Graph) tf.scope().env();
        try (Session session = new Session(graph)) {
            session.run(operand);
            print(session, prefix, operand);
        }
    }

    private static <T extends TType> void print(Session session, String prefix, Operand<T> input) {

        PrintWriter writer = new PrintWriter(System.out);
        if (input == null) {
            writer.printf("\n===================  %s  (null) ===================\n", prefix);
            return;
        }

        writer.printf("\n===================  %s (%s)  ===================\n", prefix, input.asOutput().toString());
        writer.printf("%s shape = (%s)\n", prefix, input.asOutput().shape().toString());
        DataType dtype = input.asOutput().dataType();
        boolean isScalar = input.asOutput().shape().size() == 1 || input.asOutput().shape().size() == 0;
        if (dtype == TFloat32.DTYPE) {
            AtomicInteger index = new AtomicInteger();
            try (Tensor<TFloat32> result = session.runner().fetch(input).run().get(0).expect(TFloat32.DTYPE)) {
                if (isScalar) {
                    writer.printf("    %s: %d). %f\n", prefix, index.getAndIncrement(), result.data().getFloat());
                } else {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        writer.printf("    %s: %s). %f\n", prefix, Arrays.toString(idx), f.getFloat());
                    });
                }
            }
        } else if (dtype == TFloat64.DTYPE) {
            AtomicInteger index = new AtomicInteger();

            try (Tensor<TFloat64> result = session.runner().fetch(input).run().get(0).expect(TFloat64.DTYPE)) {
                if (isScalar) {
                    writer.printf("    %s: %d). %f\n", prefix, index.getAndIncrement(), result.data().getDouble());
                } else {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        writer.printf("    %s: %s). %f\n", prefix, Arrays.toString(idx), f.getDouble());
                    });
                }
            }
        } else if (dtype == TInt32.DTYPE) {
            AtomicInteger index = new AtomicInteger();

            try (Tensor<TInt32> result = session.runner().fetch(input).run().get(0).expect(TInt32.DTYPE)) {
                if (isScalar) {
                    writer.printf("    %s: %d). %d\n", prefix, index.getAndIncrement(), result.data().getInt());
                } else {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        writer.printf("    %s: %s). %d\n", prefix, Arrays.toString(idx), f.getInt());
                    });
                }
            }
        } else if (dtype == TInt64.DTYPE) {
            AtomicInteger index = new AtomicInteger();

            try (Tensor<TInt64> result = session.runner().fetch(input).run().get(0).expect(TInt64.DTYPE)) {
                if (isScalar) {
                    writer.printf("    %s: %d). %d\n", prefix, index.getAndIncrement(), result.data().getLong());
                } else {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        writer.printf("    %s: %s). %d\n", prefix, Arrays.toString(idx), f.getLong());
                    });
                }
            }
        } else if (dtype == TBool.DTYPE) {
            AtomicInteger index = new AtomicInteger();

            try (Tensor<TBool> result = session.runner().fetch(input).run().get(0).expect(TBool.DTYPE)) {
                if (isScalar) {
                    writer.printf("    %s: %d). %b\n", prefix, index.getAndIncrement(), result.data().getBoolean());
                } else {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        writer.printf("    %s: %s). %b\n", prefix, Arrays.toString(idx), f.getBoolean());
                    });
                }
            }
        } else if (dtype == TString.DTYPE) {
            AtomicInteger index = new AtomicInteger();

            try (Tensor<TString> result = session.runner().fetch(input).run().get(0).expect(TString.DTYPE)) {
                if (isScalar) {
                    writer.printf("    %s: %d). \"%s\"\n", prefix, index.getAndIncrement(), result.data().getObject());
                } else {
                    result.data().scalars().forEachIndexed((idx, f) -> {
                        writer.printf("    %s: %s). \"%s\"\n", prefix, Arrays.toString(idx), f.getObject());
                    });
                }
            }
        } else {
            writer.println("Unexpected DataType: " + dtype);
        }
        writer.flush();
    }

    //TODO  debug, take out after unit tests are complete
    private static Session session;

    public static void setDebug(Session sess) {
        session = sess;
    }

    public static void debug(String prefix, Operand operand) {

        if (session != null) {
            print(session, prefix, operand);
        }
    }

}
