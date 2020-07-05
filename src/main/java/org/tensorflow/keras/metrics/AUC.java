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
package org.tensorflow.keras.metrics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.ControlDependencies;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.metrics.impl.ConfusionMatrixEnum;
import org.tensorflow.keras.utils.SymbolicShape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 * Computes the approximate AUC (Area under the curve) via a Riemann sum.
 *
 * @author Jim Clarke
 */
public class AUC extends Metric {
    

    public static final String TRUE_POSITIVES = "TRUE_POSITIVES";
    public static final String FALSE_POSITIVES = "FALSE_POSITIVES";
    public static final String TRUE_NEGATIVES = "TRUE_NEGATIVES";
    public static final String FALSE_NEGATIVES = "FALSE_NEGATIVES";
    public static final int DEFAULT_NUM_THRESHOLDS = 200;

    private int numThresholds;
    private AUCCurve curve;
    private AUCSummationMethod summationMethod;
    private float[] thresholds;
    private boolean multiLabel;
    private Integer numLabels;
    private Operand labelWeights;
    private Op labelWeightsChecks;

    private Variable<TFloat32> truePositives;
    private Variable<TFloat32> falsePositives;
    private Variable<TFloat32> trueNegatives;
    private Variable<TFloat32> falseNegatives;

    private boolean initialized = false;
    private Shape buildInputShape;
    
    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     */
    public AUC(Ops tf) {
        this(tf, null, DEFAULT_NUM_THRESHOLDS, AUCCurve.ROC, AUCSummationMethod.INTERPOLATION, null, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     */
    public AUC(Ops tf, String name) {
        this(tf, name, DEFAULT_NUM_THRESHOLDS, AUCCurve.ROC, AUCSummationMethod.INTERPOLATION, null, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param numThresholds the number of thresholds to use when discretizing
     * the roc curve. Values must be > 1. Defaults to 200.
     */
    public AUC(Ops tf, int numThresholds) {
        this(tf, null, numThresholds, AUCCurve.ROC, AUCSummationMethod.INTERPOLATION, null, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param thresholds Optional values to use as the thresholds for
     * discretizing the curve. If set, the numThresholds parameter is ignored.
     * Values should be in [0, 1].
     */
    public AUC(Ops tf, float[] thresholds) {
        this(tf, null, null, AUCCurve.ROC, AUCSummationMethod.INTERPOLATION, thresholds, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param numThresholds the number of thresholds to use when discretizing
     * the roc curve. Values must be > 1. Defaults to 200.
     */
    public AUC(Ops tf, String name, int numThresholds) {
        this(tf, name, numThresholds, AUCCurve.ROC, AUCSummationMethod.INTERPOLATION, null, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param thresholds Optional values to use as the thresholds for
     * discretizing the curve. If set, the numThresholds parameter is ignored.
     * Values should be in [0, 1].
     */
    public AUC(Ops tf, String name, float[] thresholds) {
        this(tf, name, null, AUCCurve.ROC, AUCSummationMethod.INTERPOLATION, thresholds, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param numThresholds the number of thresholds to use when discretizing
     * the roc curve. Values must be > 1. Defaults to 200.
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     */
    public AUC(Ops tf, String name, int numThresholds, AUCCurve curve) {
        this(tf, name, numThresholds, curve, AUCSummationMethod.INTERPOLATION, null, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param thresholds Optional values to use as the thresholds for
     * discretizing the curve. If set, the numThresholds parameter is ignored.
     * Values should be in [0, 1].
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     */
    public AUC(Ops tf, String name, float[] thresholds, AUCCurve curve) {
        this(tf, name, null, curve, AUCSummationMethod.INTERPOLATION, thresholds, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param numThresholds the number of thresholds to use when discretizing
     * the roc curve. Values must be > 1. Defaults to 200.
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     */
    public AUC(Ops tf, int numThresholds, AUCCurve curve) {
        this(tf, null, numThresholds, curve, AUCSummationMethod.INTERPOLATION, null, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param thresholds Optional values to use as the thresholds for
     * discretizing the curve. If set, the numThresholds parameter is ignored.
     * Values should be in [0, 1].
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     */
    public AUC(Ops tf, float[] thresholds, AUCCurve curve) {
        this(tf, null, null, curve, AUCSummationMethod.INTERPOLATION, thresholds, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param numThresholds the number of thresholds to use when discretizing
     * the roc curve. Values must be > 1. Defaults to 200.
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     * @param summationMethod Specifies the Riemann summation method used,
     * default is 'INTERPOLATION'
     */
    public AUC(Ops tf, int numThresholds, AUCCurve curve, AUCSummationMethod summationMethod) {
        this(tf, null, numThresholds, curve, summationMethod, null, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param thresholds Optional values to use as the thresholds for
     * discretizing the curve. If set, the numThresholds parameter is ignored.
     * Values should be in [0, 1].
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     * @param summationMethod Specifies the Riemann summation method used,
     * default is 'INTERPOLATION'
     */
    public AUC(Ops tf, float[] thresholds, AUCCurve curve, AUCSummationMethod summationMethod) {
        this(tf, null, null, curve, summationMethod, thresholds, false, null, null);
    }

    /**
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param numThresholds the number of thresholds to use when discretizing
     * the roc curve. Values must be > 1. Defaults to 200.
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     * @param summationMethod Specifies the Riemann summation method used,
     * default is 'INTERPOLATION'
     */
    public AUC(Ops tf, String name, int numThresholds, AUCCurve curve, AUCSummationMethod summationMethod) {
        this(tf, name, numThresholds, curve, summationMethod, null, false, null, null);
    }

    /**
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param thresholds Optional values to use as the thresholds for
     * discretizing the curve. If set, the numThresholds parameter is ignored.
     * Values should be in [0, 1].
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     * @param summationMethod Specifies the Riemann summation method used,
     * default is 'INTERPOLATION'
     */
    public AUC(Ops tf, String name, float[] thresholds, AUCCurve curve, AUCSummationMethod summationMethod) {
        this(tf, name, null, curve, summationMethod, thresholds, false, null, null);
    }

    /**
     * Creates an AUC (Area under the curve) metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param numThresholds the number of thresholds to use when discretizing
     * the roc curve. Values must be > 1. Defaults to 200.
     * @param curve specifies the type of the curve to be computed, 'ROC'
     * [default] or 'PR' for the Precision-Recall-curve.
     * @param summationMethod Specifies the Riemann summation method used,
     * default is 'INTERPOLATION'
     * @param thresholds Optional values to use as the thresholds for
     * discretizing the curve. If set, the numThresholds parameter is ignored.
     * Values should be in [0, 1].
     * @param multiLabel boolean indicating whether multilabel data should be
     * treated as such, wherein AUC is computed separately for each label and
     * then averaged across labels, or (when False) if the data should be
     * flattened into a single label before AUC computation. In the latter case,
     * when multilabel data is passed to AUC, each label-prediction pair is
     * treated as an individual data point. Should be set to False for
     * multi-class data.
     * @param labelWeights non-negative weights used to compute AUCs for
     * multilabel data. When multi_label is True, the weights are applied to the
     * individual label AUCs when they are averaged to produce the multi-label
     * AUC. When it's False, they are used to weight the individual label
     * predictions in computing the confusion matrix on the flattened data.
     * @param dType the type of the metric result.
     */
    public AUC(Ops tf, String name, Integer numThresholds, AUCCurve curve, AUCSummationMethod summationMethod,
            float[] thresholds, boolean multiLabel, Operand labelWeights, DataType dType) {
        super(tf, name == null ? "auc" : name, dType);

        this.curve = curve;
        this.summationMethod = summationMethod;

        this.multiLabel = multiLabel;

        if (thresholds != null) { // ignore numThresholds
            this.numThresholds = thresholds.length + 2;
            Arrays.sort(thresholds);
        } else {
            assert numThresholds > 1 : "`numThresholds` must be > 1.";
            this.numThresholds = numThresholds;
            thresholds = new float[numThresholds-2];
            //linearly interpolate (num_thresholds - 2) thresholds between endpoints
            for (int i = 0; i <thresholds.length; i++) {
                thresholds[i] = (i + 1) * 1.0f / (this.numThresholds - 1);
            }
        }
        // Add an endpoint "threshold" below zero and above one for either
        // threshold method to account for floating point imprecisions.
        assert thresholds.length == this.numThresholds-2;
        this.thresholds = new float[this.numThresholds];
        this.thresholds[0] = -K.EpsilonF;
        System.arraycopy(thresholds, 0, this.thresholds, 1, thresholds.length);
        this.thresholds[this.numThresholds - 1] = 1 + K.EpsilonF;
        
        if (labelWeights != null) {
            // assert that labelWeights are non-negative.
            labelWeights = tf.dtypes.cast(getLabelWeights(), dType);
            Op checks = tf.assertThat(tf.math.greaterEqual(labelWeights, K.zero(tf, getLabelWeights().asOutput().dataType())),
                    Arrays.asList(tf.constant("All values of `label_weights` must be non-negative."))
            );
            this.labelWeights = ControlDependencies.addControlDependencies(tf, labelWeights, 
                "updateState", checks);
        }

        if (this.multiLabel) {
            this.numLabels = null;
        } else {
            build(null);
        }
    }

    /**
     * Initialize TP, FP, TN, and FN tensors, given the shape of the data.
     *
     * @param shape the prediction shape if called from updateState, otherwise
     * null
     */
    private void build(Shape shape) {
        Shape variableShape;

        if (this.isMultiLabel()) {
            assert shape != null : "For multiLabel, a shape must be provided";
            assert shape.numDimensions() == 2 :
                    String.format("`y_true` must have rank=2 when `multi_label` is True. Found rank %d.", shape.numDimensions());
           this.numLabels = (int)shape.size(1);
           variableShape = Shape.of(this.numThresholds, this.numLabels);
        } else {
            variableShape =  Shape.of(this.numThresholds);
        }

        this.buildInputShape = shape;
        
        Zeros zeros = new Zeros(tf);
        
        truePositives = getVariable(TRUE_POSITIVES);
        if (truePositives == null) {
            
            truePositives = tf.withName(TRUE_POSITIVES).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(TRUE_POSITIVES, truePositives, zeros);
        }
        falsePositives = getVariable(FALSE_POSITIVES);
        if (falsePositives == null) {
            falsePositives = tf.withName(FALSE_POSITIVES).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(FALSE_POSITIVES, falsePositives, zeros);
        }
        trueNegatives = getVariable(TRUE_NEGATIVES);
        if (trueNegatives == null) {
            trueNegatives = tf.withName(TRUE_NEGATIVES).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(TRUE_NEGATIVES, trueNegatives, zeros);
        }
        falseNegatives = getVariable(FALSE_NEGATIVES);
        if (falseNegatives == null) {
            falseNegatives = tf.withName(FALSE_NEGATIVES).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(FALSE_NEGATIVES, falseNegatives, zeros);
        }

        this.initialized = true;
    }
    
    
    


    /**
     * {@inheritDoc}
     */
    @Override
    public Op updateState(Operand... args) {
        Operand yTrue = args[0];
        Operand yPred = args[1];
        Operand sampleWeights = args.length > 2 ? args[2] : null;
        List<Op> updateOperations = new ArrayList<>();

        if (!this.initialized) {
            build(yPred.asOutput().shape());
        }
        if (this.isMultiLabel() || this.getLabelWeights() != null) {
            List<SymbolicShape> symbols = new ArrayList<>();
            symbols.add(new SymbolicShape(yTrue, "N", "L"));
            if (this.isMultiLabel()) {
                symbols.add(new SymbolicShape(this.truePositives, "T", "L"));
                symbols.add(new SymbolicShape(this.falsePositives, "T", "L"));
                symbols.add(new SymbolicShape(this.trueNegatives, "T", "L"));
                symbols.add(new SymbolicShape(this.falseNegatives, "T", "L"));
            }
            if (this.getLabelWeights() != null) {
                symbols.add(new SymbolicShape(this.getLabelWeights(), "L", ""));
            }
            updateOperations.addAll(Metrics.assert_shapes(tf,
                    symbols, "Number of labels is not consistent."));

        }
        if (this.isMultiLabel()) {
            this.labelWeights = null;
        }
        Map<ConfusionMatrixEnum, Variable> confusionMatrix = new HashMap<>();
        confusionMatrix.put(ConfusionMatrixEnum.TRUE_POSITIVES, this.truePositives);
        confusionMatrix.put(ConfusionMatrixEnum.FALSE_POSITIVES, this.falsePositives);
        confusionMatrix.put(ConfusionMatrixEnum.TRUE_NEGATIVES, this.trueNegatives);
        confusionMatrix.put(ConfusionMatrixEnum.FALSE_NEGATIVES, this.falseNegatives);

        updateOperations.addAll(Metrics.update_confusion_matrix_variables(tf,
                confusionMatrix,
                yTrue,
                yPred, this.thresholds,
                null,
                null,
                sampleWeights, this.isMultiLabel(), this.getLabelWeights()));
        return ControlDependencies.addControlDependencies(tf,
                "updateState", updateOperations);
    }

    /**
     * Interpolation formula inspired by section 4 of Davis & Goadrich 2006.
     *
     * @return an approximation of the area under the P-R curve.
     */
    private Operand interpolatePRAuc() {
        // true_positives[:self.num_thresholds - 1]
        Operand tp_0 =  tf.slice(truePositives, 
                tf.constant(new int[]{0}),
                tf.constant(new int[]{this.getNumThresholds() - 1}));
        // true_positives[1:]
        Operand tp_1 =  tf.slice(truePositives, 
                tf.constant(new int[]{1}),
                tf.constant(new int[]{-1}));

        //TODO remove MetricsImpl.debug("tp_0", tp_0);
        //TODO remove MetricsImpl.debug("tp_1", tp_1);
        
        Operand dTP = tf.math.sub(tp_0, tp_1);
        
        //TODO remove MetricsImpl.debug("dtp", dTP);
        
        Operand p = tf.math.add(truePositives, falsePositives);
        //TODO remove MetricsImpl.debug("p", p);
        
        Operand dP = tf.math.sub(
            tf.slice(p, tf.constant(new int[]{0}), tf.constant(new int[]{this.getNumThresholds() - 1})),
            tf.slice(p, tf.constant(new int[]{1}), tf.constant(new int[]{-1}))
        );
        //TODO remove MetricsImpl.debug("dP", dP);

        Operand precSlope = tf.math.divNoNan(
                dTP, tf.math.maximum(dP, tf.dtypes.cast(tf.constant(0), dP.asOutput().dataType())));

        //TODO remove MetricsImpl.debug("precSlope", precSlope);
        
        Operand intercept = tf.math.sub(tf.slice(truePositives, tf.constant(new int[]{1}),
                tf.constant(new int[]{-1})),
                tf.math.mul(precSlope,
                        tf.slice(p, tf.constant(new int[]{1}), tf.constant(new int[]{-1})))
        );
        
        //TODO remove MetricsImpl.debug("intercept", intercept);

        Operand safePRatio = tf.select(tf.math.logicalAnd(tf.math.greater(tf.slice(p, tf.constant(new int[]{0}), tf.constant(new int[]{this.getNumThresholds() - 1})),
                tf.dtypes.cast(tf.constant(0), p.asOutput().dataType())),
                tf.math.greater(
                        tf.slice(p, tf.constant(new int[]{1}), tf.constant(new int[]{-1})),
                        tf.dtypes.cast(tf.constant(0), p.asOutput().dataType()))
        ),
                tf.math.divNoNan(tf.slice(p, tf.constant(new int[]{0}), tf.constant(new int[]{this.getNumThresholds() - 1})),
                        tf.math.maximum(
                                tf.slice(p, tf.constant(new int[]{1}), tf.constant(new int[]{-1})),
                                tf.dtypes.cast(tf.constant(0), p.asOutput().dataType()))
                ),
                tf.onesLike(tf.slice(p, tf.constant(new int[]{1}), tf.constant(new int[]{-1})))
        );
        
        //TODO remove MetricsImpl.debug("safePRatio", safePRatio);
        Operand fn_1 = tf.slice(falseNegatives,  tf.constant(new int[]{1}), tf.constant(new int[]{-1}));
        //TODO remove MetricsImpl.debug("fn_1", fn_1);
        
        Operand  auc_TotalPos = tf.math.mul(precSlope, tf.math.add(dTP, tf.math.mul(intercept, tf.math.log(safePRatio))));
        //TODO remove MetricsImpl.debug("auc_TotalPos", auc_TotalPos);
        
        Operand pr_auc_increment = tf.math.divNoNan(
                auc_TotalPos,
                tf.math.maximum(tf.math.add( tp_1, fn_1), 
                        tf.dtypes.cast(tf.constant(0), this.truePositives.asOutput().dataType()))
        );
        //TODO remove MetricsImpl.debug("pr_auc_increment", pr_auc_increment);
        if (this.isMultiLabel()) {
            Operand by_label_auc = tf.reduceSum(pr_auc_increment, tf.constant(0));
            if (this.getLabelWeights() == null) {
                return K.mean(tf, by_label_auc);
            } else {
                return tf.math.divNoNan(tf.reduceSum(tf.math.mul(by_label_auc, this.getLabelWeights()),
                        K.allAxis(tf, by_label_auc)),
                        tf.reduceSum(getLabelWeights(), K.allAxis(tf, getLabelWeights()))
                );
            }
        } else {
            return tf.reduceSum(pr_auc_increment, K.allAxis(tf, pr_auc_increment));
        }

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand result() {
                
        //TODO remove MetricsImpl.debug("result()/TRUE_POSITIVES", truePositives);
        //TODO remove MetricsImpl.debug("result()/FALSE_POSITIVES", falsePositives);
        //TODO remove MetricsImpl.debug("result()/TRUE_NEGATIVES", trueNegatives);
        //TODO remove MetricsImpl.debug("result()/FALSE_NEGATIVES", falseNegatives);
         
        if (this.getCurve() == AUCCurve.PR && this.getSummationMethod() == AUCSummationMethod.INTERPOLATION) {
            return this.interpolatePRAuc();
        }
        Operand x;
        Operand y;
        Operand recall = tf.math.divNoNan(truePositives, tf.math.add(truePositives, falseNegatives));


        if (this.getCurve() == AUCCurve.ROC) {
            Operand fpRate = tf.math.divNoNan(falsePositives, tf.math.add(falsePositives, trueNegatives));
            x = fpRate;
            y = recall;
             //TODO remove MetricsImpl.debug("result()//x(fpRate)", x);
        } else { //AUCCurve.PR
            Operand precision = tf.math.divNoNan(truePositives, tf.math.add(truePositives, falsePositives));
            y = precision;
            x = recall;
             //TODO remove MetricsImpl.debug("result()//x(precision)", x);
        }
        
       
        //TODO remove MetricsImpl.debug("result()//y(recall)", y);
        

        // Find the rectangle heights based on `summation_method`.
        //y[:self.num_thresholds - 1]
        Operand ySlice1 = tf.slice(y, tf.constant(new int[]{0}),
                tf.constant(new int[]{this.getNumThresholds() - 1}));
        //y[1:]
        Operand ySlice2 = tf.slice(y, tf.constant(new int[]{1}), 
                tf.constant(new int[]{-1}));
        
        //TODO remove MetricsImpl.debug("result()/ySlice1", ySlice1);
        //TODO remove MetricsImpl.debug("result()/ySlice2", ySlice2);
        Operand heights = null;
        switch (this.getSummationMethod()) {
            case INTERPOLATION:
                heights = tf.math.div(
                        tf.math.add(ySlice1, ySlice2),
                        tf.dtypes.cast(tf.constant(2), y.asOutput().dataType())
                );
                break;
            case MINORING:
                heights = tf.math.minimum(ySlice1, ySlice2);
                break;
            case MAJORING:
                heights = tf.math.maximum(ySlice1, ySlice2);
                break;
        }
        //TODO remove MetricsImpl.debug("AUC/heights", heights);

        if (this.isMultiLabel()) {
            Operand riemann_terms = tf.math.mul(tf.math.sub(tf.slice(x, tf.constant(new int[]{0}), tf.constant(new int[]{this.getNumThresholds() - 1})),
                    tf.slice(x, tf.constant(new int[]{1}), tf.constant(new int[]{-1})) ),
                    heights );
            Operand by_label_auc = tf.reduceSum(riemann_terms, tf.constant(0));
            
            if (this.getLabelWeights() == null) {
                return K.mean(tf, by_label_auc);
            } else {
                return tf.math.divNoNan(tf.reduceSum(tf.math.mul(by_label_auc, getLabelWeights()), K.allAxis(tf, getLabelWeights())),
                        tf.reduceSum(getLabelWeights(), K.allAxis(tf, getLabelWeights()))
                );
            }

        } else {
            Operand slice1 = tf.slice(x, tf.constant(new int[]{0}), tf.constant(new int[]{this.getNumThresholds() - 1}));
            //TODO remove MetricsImpl.debug("AUC/slice1", slice1);
            Operand slice2 = tf.slice(x, tf.constant(new int[]{1}), tf.constant(new int[]{-1}));
            //TODO remove MetricsImpl.debug("AUC/slice2", slice2);
            Operand sub = tf.math.sub(slice1, slice2);
            //TODO remove MetricsImpl.debug("AUC/sub", sub);
            Operand operand =  tf.math.mul( sub, heights);
            //TODO remove MetricsImpl.debug("result()//operand", operand);
            Operand sum =  tf.reduceSum(operand,  K.allAxis(tf, operand) );
            //TODO remove MetricsImpl.debug("result()//reducesum", sum);
            return sum;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Op resetStates() {
        List<Op> updateOperations = new ArrayList<>();
        if (isMultiLabel()) {
            final Shape varShape = Shape.of(this.getNumThresholds(), this.getNumLabels());
            variables.values().forEach((v)
                    -> updateOperations.add(tf.assign(v.getVariable(),
                            tf.zeros(tf.constant(varShape), v.getVariable().asOutput().dataType())))
            );
        } else {
            final Shape varShape = Shape.of(this.getNumThresholds());
            variables.values().forEach((v)
                    -> updateOperations.add(tf.assign(v.getVariable(),
                            tf.zeros(tf.constant(varShape), v.getVariable().asOutput().dataType())))
            );
        }
        return ControlDependencies.addControlDependencies(tf, "resetStates", updateOperations);
    }

    /**
     * @return the numThresholds
     */
    public int getNumThresholds() {
        return numThresholds;
    }

    /**
     * @return the curve
     */
    public AUCCurve getCurve() {
        return curve;
    }

    /**
     * @return the summationMethod
     */
    public AUCSummationMethod getSummationMethod() {
        return summationMethod;
    }

    /**
     * @return the thresholds
     */
    public float[] getThresholds() {
        return thresholds;
    }

    /**
     * @return the multiLabel
     */
    public boolean isMultiLabel() {
        return multiLabel;
    }

    /**
     * @return the numLabels
     */
    public Integer getNumLabels() {
        return numLabels;
    }

    /**
     * @param numLabels the numLabels to set
     */
    public void setNumLabels(Integer numLabels) {
        this.numLabels = numLabels;
    }

    /**
     * @return the labelWeights
     */
    public Operand getLabelWeights() {
        return labelWeights;
    }

    /**
     * @return the truePositives
     */
    public Variable<TFloat32> getTruePositives() {
        return truePositives;
    }

    /**
     * @return the falsePositives
     */
    public Variable<TFloat32> getFalsePositives() {
        return falsePositives;
    }

    /**
     * @return the trueNegatives
     */
    public Variable<TFloat32> getTrueNegatives() {
        return trueNegatives;
    }

    /**
     * @return the falseNegatives
     */
    public Variable<TFloat32> getFalseNegatives() {
        return falseNegatives;
    }

}
