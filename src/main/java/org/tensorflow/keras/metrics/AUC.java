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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.tf.ControlDependencies;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.metrics.impl.ConfusionMatrixEnum;
import static org.tensorflow.keras.metrics.impl.SensitivitySpecificityBase.FALSE_NEGATIVES;
import static org.tensorflow.keras.metrics.impl.SensitivitySpecificityBase.FALSE_POSITIVES;
import static org.tensorflow.keras.metrics.impl.SensitivitySpecificityBase.TRUE_NEGATIVES;
import static org.tensorflow.keras.metrics.impl.SensitivitySpecificityBase.TRUE_POSITIVES;
import org.tensorflow.keras.utils.SymbolicShape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
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
    private final String truePositivesName;
    private final String falsePositivesName;
    private final String trueNegativesName;
    private final String falseNegativesName;

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
        this.truePositivesName = this.getClass().getSimpleName() + "_" + TRUE_POSITIVES;
        this.falsePositivesName = this.getClass().getSimpleName() + "_" + FALSE_POSITIVES;
        this.trueNegativesName = this.getClass().getSimpleName() + "_" + TRUE_NEGATIVES;
        this.falseNegativesName = this.getClass().getSimpleName() + "_" + FALSE_NEGATIVES;
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
            
            this.labelWeights  = tf.dtypes.cast(getLabelWeights(), dType);
            Op checks = tf.assertThat(tf.math.greaterEqual(labelWeights, K.zero(tf, getLabelWeights().asOutput().dataType())),
                    Arrays.asList(tf.constant("All values of `label_weights` must be non-negative."))
            );
            
            this.labelWeights = ControlDependencies.addControlDependencies(tf, 
                    tfc -> tfc.dtypes.cast(getLabelWeights(), dType),
                    "updateState", checks);
        }

        if (this.multiLabel) {
            this.numLabels = null;
        //} else {
        //    build(null);
        }
    }

    /**
     * Initialize TP, FP, TN, and FN tensors, given the shape of the data.
     *
     * @param shape the prediction shape if called from updateState, otherwise
     * null
     */
    private Map<ConfusionMatrixEnum, Assign>  build(Shape shape) {
        Shape variableShape;
        if(initialized) return Collections.EMPTY_MAP;
        Map<ConfusionMatrixEnum, Assign> initializers = new HashMap<>();
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
        
        truePositives = getVariable(getTruePositivesName());
        if (truePositives == null) {
            truePositives = tf.withName(getTruePositivesName()).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(getTruePositivesName(), truePositives, zeros);
            initializers.put(ConfusionMatrixEnum.TRUE_POSITIVES,
                tf.assign(truePositives, zeros.call(tf.constant(variableShape), TFloat32.DTYPE)));
            
        }
        falsePositives = getVariable(getFalsePositivesName());
        if (falsePositives == null) {
            falsePositives = tf.withName(getFalsePositivesName()).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(getFalsePositivesName(), falsePositives, zeros);
            initializers.put(ConfusionMatrixEnum.FALSE_POSITIVES,
                tf.assign(falsePositives, zeros.call(tf.constant(variableShape), TFloat32.DTYPE)));
        }
        trueNegatives = getVariable(getTrueNegativesName());
        if (trueNegatives == null) {
            trueNegatives = tf.withName(getTrueNegativesName()).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(getTrueNegativesName(), trueNegatives, zeros);
            initializers.put(ConfusionMatrixEnum.TRUE_NEGATIVES,
                tf.assign(trueNegatives, zeros.call(tf.constant(variableShape), TFloat32.DTYPE)));
        }
        falseNegatives = getVariable(getFalseNegativesName());
        if (falseNegatives == null) {
            falseNegatives = tf.withName(getFalseNegativesName()).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(getFalseNegativesName(), falseNegatives, zeros);
            initializers.put(ConfusionMatrixEnum.FALSE_NEGATIVES,
                tf.assign(falseNegatives, zeros.call(tf.constant(variableShape), TFloat32.DTYPE)));
        }

        this.initialized = true;
        return initializers;
    }
    
    
    


    /**
     * {@inheritDoc}
     */
    @Override
    public List<Op> updateStateList(Operand... args) {
        Operand yTrue = args[0];
        Operand yPred = args[1];
        Operand sampleWeights = args.length > 2 ? args[2] : null;
        List<Op> updateOperations = new ArrayList<>();
        Map<ConfusionMatrixEnum, Assign> varInitalizers = Collections.EMPTY_MAP;
        if (!this.initialized) {
            varInitalizers = build(yPred.asOutput().shape());
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
                varInitalizers,
                yTrue,
                yPred, this.thresholds,
                null,
                null,
                sampleWeights, this.isMultiLabel(), this.getLabelWeights()));
        return updateOperations;
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
    public Operand result(Ops rtf) {
                
         
        if (this.getCurve() == AUCCurve.PR && this.getSummationMethod() == AUCSummationMethod.INTERPOLATION) {
            return this.interpolatePRAuc();
        }
        Operand x;
        Operand y;
        Operand recall = rtf.math.divNoNan(truePositives, rtf.math.add(truePositives, falseNegatives));


        if (this.getCurve() == AUCCurve.ROC) {
            Operand fpRate = rtf.math.divNoNan(falsePositives, rtf.math.add(falsePositives, trueNegatives));
            x = fpRate;
            y = recall;
        } else { //AUCCurve.PR
            Operand precision = rtf.math.divNoNan(truePositives, rtf.math.add(truePositives, falsePositives));
            y = precision;
            x = recall;
        }
        
       

        // Find the rectangle heights based on `summation_method`.
        //y[:self.num_thresholds - 1]
        Operand ySlice1 = rtf.slice(y, rtf.constant(new int[]{0}),
                rtf.constant(new int[]{this.getNumThresholds() - 1}));
        //y[1:]
        Operand ySlice2 = rtf.slice(y, rtf.constant(new int[]{1}), 
                rtf.constant(new int[]{-1}));
        
        Operand heights = null;
        switch (this.getSummationMethod()) {
            case INTERPOLATION:
                heights = rtf.math.div(
                        rtf.math.add(ySlice1, ySlice2),
                        rtf.dtypes.cast(rtf.constant(2), y.asOutput().dataType())
                );
                break;
            case MINORING:
                heights = rtf.math.minimum(ySlice1, ySlice2);
                break;
            case MAJORING:
                heights = rtf.math.maximum(ySlice1, ySlice2);
                break;
        }
        //TODO remove MetricsImpl.debug("AUC/heights", heights);

        if (this.isMultiLabel()) {
            Operand riemann_terms = rtf.math.mul(rtf.math.sub(rtf.slice(x, rtf.constant(new int[]{0}), rtf.constant(new int[]{this.getNumThresholds() - 1})),
                    rtf.slice(x, rtf.constant(new int[]{1}), rtf.constant(new int[]{-1})) ),
                    heights );
            Operand by_label_auc = rtf.reduceSum(riemann_terms, rtf.constant(0));
            
            if (this.getLabelWeights() == null) {
                return K.mean(rtf, by_label_auc);
            } else {
                return rtf.math.divNoNan(rtf.reduceSum(rtf.math.mul(by_label_auc, getLabelWeights()), K.allAxis(rtf, getLabelWeights())),
                        rtf.reduceSum(getLabelWeights(), K.allAxis(rtf, getLabelWeights()))
                );
            }

        } else {
            Operand slice1 = rtf.slice(x, rtf.constant(new int[]{0}), rtf.constant(new int[]{this.getNumThresholds() - 1}));
            Operand slice2 = rtf.slice(x, rtf.constant(new int[]{1}), rtf.constant(new int[]{-1}));
            Operand sub = rtf.math.sub(slice1, slice2);
            Operand operand =  rtf.math.mul( sub, heights);
            Operand sum =  rtf.reduceSum(operand,  K.allAxis(rtf, operand) );
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
            this.getVariables().forEach((v)
                    -> updateOperations.add(tf.assign(v,
                            tf.zeros(tf.constant(varShape), v.asOutput().dataType())))
            );
        } else {
            final Shape varShape = Shape.of(this.getNumThresholds());
            this.getVariables().forEach((v)
                    -> updateOperations.add(tf.assign(v,
                            tf.zeros(tf.constant(varShape), v.asOutput().dataType())))
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

    /**
     * @return the truePositivesName
     */
    public String getTruePositivesName() {
        return truePositivesName;
    }

    /**
     * @return the falsePositivesName
     */
    public String getFalsePositivesName() {
        return falsePositivesName;
    }

    /**
     * @return the trueNegativesName
     */
    public String getTrueNegativesName() {
        return trueNegativesName;
    }

    /**
     * @return the falseNegativesName
     */
    public String getFalseNegativesName() {
        return falseNegativesName;
    }

}
